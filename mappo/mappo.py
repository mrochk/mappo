import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from pettingzoo import AECEnv
import os

from .critic import Critic
from .actor import Actor

class MAPPO(nn.Module):
    '''Multi Agent Proximal Policy Optimization

    Centralized Training - Decentralized Execution Framework: 1 Critic for N Actors.
    '''

    def __init__(
        self, 
        env: AECEnv,
        batch_size: int = 64,
        epochs: int = 10,
        eps: float = 0.2,
        c_ent: float = 0.0,
    ):
        super().__init__()

        self.env = env; self.env.reset()
        state_space_size = np.prod(env.observation_space(env.agents[0]).shape)

        self.actors = nn.ModuleDict({
            agent: Actor(
                state_space_size=state_space_size,
                n_actions=env.action_space(agent).n,
                batch_size=batch_size,
                eps=eps,
                epochs=epochs,
                c_ent=c_ent,
            )
            for agent in env.agents
        })

        self.centralized_critic = Critic(state_space_size, batch_size, 20)

    def learn(
        self,
        niters: int,
        nsteps: int = 2048,
        checkpoint_path: str = 'checkpoints_mappo',
        eval_path: str = 'evaluations_mappo',
    ):
        '''Train the algorithm for `niters` iterations.

        Args:
            niters (int): How many updates to perform on actors and critic.
            nsteps (int, optional): How many steps to collect before performing an update. Defaults to 2048.
            checkpoint_path (str, optional): Path to save algorithm state. Defaults to 'checkpoints'.
        '''

        try: os.mkdir(checkpoint_path)
        except: pass

        evaluations = {'mean_return': [], 'std_return': [], 'mean_eplen': []}

        for i in range(niters):

            # collect trajectories
            trajectories = self.collect_trajectories(nsteps)
            trajectories = self.split_trajectories(trajectories)

            # compute GAE and returns
            trajectories = self.centralized_critic.add_gae(trajectories, gamma=0.99, lam=0.95)

            # flatten trajectories before training
            flattened = self.flatten_trajectories(trajectories)

            # updates critic and actors
            loss_critic = self.centralized_critic.update(flattened)
            self.update_actors(flattened)

            # eval
            with th.no_grad():
                mean_return, std_return, mean_length = self.evaluate(10)
                evaluations['mean_return'].append(mean_return)
                evaluations['mean_eplen'].append(mean_length)
                evaluations['std_return'].append(std_return)

                print(f'¤ {i+1} / {niters}: Reward={mean_return:.1f}, Std={std_return:.1f}, Length={mean_length:.0f}, LossCritic={loss_critic:.3f}')

            # save checkpoint
            th.save(self.state_dict(), f'{checkpoint_path}/mappo{i+1}_{mean_return:.1f}_{std_return:.1f}.pth')

            # save for plotting
            np.save(eval_path, np.array(evaluations))

    def flatten_trajectories(self, trajectories: dict):
        '''Flatten the collected trajectories.

        `collect_trajectories` returns a dict with, for each agent, N collected trajectories. 
        This function flatten the N trajectories into one "big" trajectory, such that we can then sample minbatches.
        '''
        flattened = {}
        for agent, trajectories in trajectories.items():
            flattened[agent] = {
                'states': th.cat([t['states'] for t in trajectories]),
                'actions': th.cat([t['actions'] for t in trajectories]),
                'other_actions': th.cat([t['other_actions'] for t in trajectories]),
                'rewards': th.cat([t['rewards'] for t in trajectories]),
                'logprobs': th.cat([t['logprobs'] for t in trajectories]),
                'advantages': th.cat([t['advantages'] for t in trajectories]),
                'returns': th.cat([t['rtgs'] for t in trajectories]),
            }

        return flattened

    def update_actors(self, flattened: dict):
        '''Update the actors one by one by calling `actor.update` using their own collected data.'''
        for agent, actor in self.actors.items():
            dataset_agent = flattened[agent]
            actor.update(dataset_agent)

    def collect_trajectories(self, N: int) -> dict:
        '''Collect `N` trajectories for each agent in the environment.
        Trajectory = (state1, action1, reward1, logprob1, state2, ...).
        '''
        trajectories = {agent: [] for agent in self.actors.keys()}
        total_length = 0
        n_agents = len(self.actors.keys())

        while total_length < N:
            for t in trajectories.values(): t.append([])
            self.env.reset()

            other_action = -1 # action of the other agent

            for agent in self.env.agent_iter():
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
                elements = [obs, action, other_action, reward, logprob]
                trajectories[agent][-1].extend(elements)
                # take a step to get next state
                self.env.step(action)

                total_length += 1 / n_agents
                other_action = action

        return trajectories 

    def split_trajectories(self, trajectories: dict) -> dict:
        return {
            agent: [self.split_trajectory(trajectory) for trajectory in trajs]
            for agent, trajs in trajectories.items()
        }

    def split_trajectory(self, trajectory: list) -> dict[str, th.Tensor]:
        '''Split a trajectory into separate tensors of same elements.'''

        states = []; rewards = []; actions = []; other_actions= []; logprobs = []
        for i in range(0, len(trajectory), 5): # elements = [obs, action, other_action, reward, logprob]
            s, a, o, r, l = trajectory[i:i+5]
            states.append(s); actions.append(a); other_actions.append(o); rewards.append(r); logprobs.append(l)

        return {
            'states': th.as_tensor(np.array(states), dtype=th.float32),
            'rewards': th.as_tensor(rewards, dtype=th.float32),
            'actions': th.as_tensor(actions, dtype=th.int64),
            'other_actions': th.as_tensor(other_actions, dtype=th.int64),
            'logprobs': th.as_tensor(logprobs, dtype=th.float32),
        }

    def evaluate(self, N: int = 10):
        '''Evaluate the algorithm for `N` episodes.'''

        with th.no_grad():
            self.eval()
            n_agents = len(self.actors.keys())
            seeds = list(range(N))
            returns, lengths = [], []

            for i in range(N):
                self.env.reset(seed=seeds[i])
                return_ = length = 0.0

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
                    length += 1 / n_agents

                returns.append(return_); lengths.append(length)

            self.train()
            return np.mean(returns), np.std(returns), np.mean(lengths)