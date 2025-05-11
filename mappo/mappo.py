import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from pettingzoo import AECEnv

from env import create_environment

from .critic import Critic
from .actor import Actor

import time

class MAPPO(nn.Module):
    env: AECEnv

    def __init__(self):
        super().__init__()

        env = create_environment(2, 2500)
        self.env = env
        self.env.reset()
        state_space_size = np.prod(env.observation_space(env.agents[0]).shape)

        self.actors = nn.ModuleDict({
            agent: Actor(
                state_space_size=state_space_size,
                n_actions=env.action_space(agent).n,
            )
            for agent in env.agents
        })

        self.centralized_critic = Critic(state_space_size)

    def learn(
        self,
        n_iters: int,
        total_steps: int,
        batch_size = 5_000,
        eps: float = 0.2,
        c_ent: float = 0.01,
        epochs_actor: int = 10,
        epochs_critic: int = 5,
        checkpoint_path: str = 'checkpoints',
    ):

        iters_count = tqdm(range(n_iters), colour='yellow')
        for i in iters_count:
            trajectories = self.collect_trajectories(total_steps)
            trajectories = self.split_trajectories(trajectories)

            #print(len(trajectories['archer_0']))
            #print(len(trajectories['archer_0'][0]['actions']))

            trajectories = self.centralized_critic.add_gae(trajectories, gamma=0.99, lam=0.95)

            flattened = self.flatten_trajectories(trajectories)

            self.centralized_critic.update(flattened, epochs=epochs_critic, batch_size=batch_size)
            self.update_actors(flattened, epochs=epochs_actor, eps=eps, c_entropy=c_ent, batch_size=batch_size)

            # eval
            with th.no_grad():
                mean_return, std_return, mean_length = self.evaluate(10)
                iters_count.set_postfix_str(
                    f'mean(returns)={mean_return:.2f} | ' +
                    f'std(returns)={std_return:.2f} | '   +
                    f'mean(length)={mean_length:.2f}'
                )

            # save checkpoint
            th.save(self.state_dict(), f'{checkpoint_path}/mappo{i+1}_{mean_return:.1f}_{std_return:.1f}.pth')
            print()

    def flatten_trajectories(self, trajectories: dict):
        flattened = {}
        for agent in self.actors.keys():
            trajs_agent = trajectories[agent]
            flattened[agent] = {
                'states': th.cat([t['states'] for t in trajs_agent]),
                'actions': th.cat([t['actions'] for t in trajs_agent]),
                'rewards': th.cat([t['rewards'] for t in trajs_agent]),
                'logprobs': th.cat([t['logprobs'] for t in trajs_agent]),
                'advantages': th.cat([t['advantages'] for t in trajs_agent]),
                'returns': th.cat([t['rtgs'] for t in trajs_agent]),
            }
        return flattened

    def update_actors(self, flattened: dict, epochs: int, eps: float, c_entropy: float, batch_size: int):
        for agent, actor in self.actors.items():
            dataset_agent = flattened[agent]
            actor.update(dataset_agent, epochs, eps, c_entropy, batch_size)

    def collect_trajectories(self, N: int) -> dict:
        '''Collect `N` trajectories for each agent in the environment.
        Trajectory = (state1, action1, reward1, logprob1, state2, ...).
        '''
        trajectories = {agent: [] for agent in self.actors.keys()}

        total_length = 0
        while True:
            for x in trajectories.values(): x.append([])

            self.env.reset()

            t = 0

            for agent in self.env.agent_iter():
                t += 1
                obs, reward, term, trunc, _ = self.env.last()

                if term or trunc:
                    # if agent is dead or max_cycles is reached
                    action = None
                    self.env.step(action)
                    continue

                # get the corresponding actor
                actor = self.actors[agent]

                # get the action, logprob pair
                action, logprob = actor.action(obs)

                # add the items to the current trajectory
                elements = [obs, action, reward, logprob]
                trajectories[agent][-1].extend(elements)

                total_length += 0.5

                # take a step to get next state
                self.env.step(action)

            if total_length > N:
                return trajectories

            print(f'[{int(total_length)}]', end=' ', flush=True)

        return trajectories

    def split_trajectories(self, trajectories: dict) -> dict:
        return {
            agent: [self.split_trajectory(trajectory) for trajectory in trajs]
            for agent, trajs in trajectories.items()
        }

    def split_trajectory(self, trajectory: list) -> dict[str, th.Tensor]:
        '''Split a trajectory into separate tensors of same elements.'''

        states = []; rewards = []; actions = []; logprobs = []
        for i in range(0, len(trajectory), 4):
            s, a, r, l = trajectory[i], trajectory[i+1], trajectory[i+2], trajectory[i+3]
            states.append(s); rewards.append(r); actions.append(a); logprobs.append(l)

        return {
            'states':   th.as_tensor(states,   dtype=th.float32),
            'rewards':  th.as_tensor(rewards,  dtype=th.float32),
            'actions':  th.as_tensor(actions,  dtype=th.int64),
            'logprobs': th.as_tensor(logprobs, dtype=th.float32),
        }

    def evaluate(self, N: int = 10):
        seeds = list(range(N))
        with th.no_grad():
            self.eval()

            returns = []
            lengths = []

            for i in range(N):
                self.env.reset(seed=seeds[i])
                return_ = 0

                length = 0.0

                for agent in self.env.agent_iter():
                    obs, reward, term, trunc, _ = self.env.last()
                    return_ += reward

                    if term or trunc:
                        # if agent is dead or max_cycles is reached
                        action = None
                        self.env.step(action)
                        continue

                    # get the corresponding actor
                    actor = self.actors[agent]
                    # get the action
                    action, _ = actor.action(obs)
                    # take a step to get next state
                    self.env.step(action)
                    length += 0.5

                returns.append(return_)
                lengths.append(length)

            self.train()

            return np.mean(returns), np.std(returns), np.mean(lengths)