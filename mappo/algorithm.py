import numpy as np
import torch as th
from torch import Tensor
from torch import nn
from pettingzoo import AECEnv
import abc, os

from .actor import Actor

class Algorithm(nn.Module, abc.ABC):
    '''Base Class for IPPO & MAPPO.'''

    def __init__(self, env: AECEnv, batch_size: int = 64, epochs: int = 10, eps: float = 0.2, c_ent: float = 0):
        '''
        Args:
            env (AECEnv): _description_
            batch_size (int, optional): _description_. Defaults to 64.
            epochs (int, optional): _description_. Defaults to 10.
            eps (float, optional): _description_. Defaults to 0.2.
            c_ent (float, optional): _description_. Defaults to 0.
        '''
        super().__init__()

        env.reset()
        self.env = env

        self.ssize = np.prod(env.observation_space(env.agents[0]).shape)

        self.actors = nn.ModuleDict({
            agent: Actor(
                statesize=self.ssize,
                n_actions=env.action_space(agent).n,
                batch_size=batch_size,
                eps=eps,
                epochs=epochs,
                c_ent=c_ent,
            )
            for agent in env.agents
        })

    def learn(self, niters: int, nsteps: int = 2048, checkpoints_path: str = 'checkpoints',
                eval_path: str = 'evaluations'):
        '''Train the algorithm for `niters` iterations.

        Args:
            niters (int): How many updates to perform.
            nsteps (int, optional): How many steps to collect before performing an update. Defaults to 2048.
            checkpoints_path (str, optional): Path to save algorithm state. Defaults to 'checkpoints'.
        '''

        # create directory for checkpoints if not exists
        try: os.mkdir(checkpoints_path)
        except: pass

        evaluations = {'mean_return': [], 'std_return': [], 'mean_eplen': []}

        for i in range(niters):

            # collect trajectories
            trajectories = self.collect_trajectories(nsteps)
            trajectories = self.split_trajectories(trajectories)

            # compute GAE and returns
            trajectories = self.add_gae(trajectories, gamma=0.99, lam=0.95)

            # flatten trajectories before training
            flattened = self.flatten_trajectories(trajectories)

            # updates critic and actors
            loss_critics = self.update_critic(flattened)
            self.update_actors(flattened)

            # eval
            with th.no_grad():
                mean_return, std_return, mean_length = self.evaluate(10)
                evaluations['mean_return'].append(mean_return)
                evaluations['mean_eplen'].append(mean_length)
                evaluations['std_return'].append(std_return)

                log = f'¤ {i+1} / {niters}: Reward={mean_return:.1f}, ' + \
                      f'Std={std_return:.1f}, Length={mean_length:.0f}, LossCritic={loss_critics:.3f}'

                print(log, flush=True)

            # save checkpoint
            th.save(self.state_dict(), f'{checkpoints_path}/ippo{i+1}_{mean_return:.1f}_{std_return:.1f}.pth')

            # save logs
            np.save(eval_path, np.array(evaluations))

    def update_actors(self, flattened: dict):
        '''Update the actors one by one by calling `actor.update` using their own collected data.'''
        for agent, actor in self.actors.items():
            dataset_agent = flattened[agent]
            actor.update(dataset_agent)

    def split_trajectories(self, trajectories: dict) -> dict:
        return {
            agent: [self.split_trajectory(trajectory) for trajectory in trajs]
            for agent, trajs in trajectories.items()
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

    @abc.abstractmethod
    def collect_trajectories(self, N: int) -> dict:
        '''Collect `N` trajectories for each agent in the environment.
        Trajectory = (state1, action1, reward1, logprob1, state2, ...).
        '''
        pass

    @abc.abstractmethod
    def split_trajectory(self, trajectory: list) -> dict[str, Tensor]:
        '''Split a trajectory into separate tensors of same elements.'''
        pass

    @abc.abstractmethod
    def flatten_trajectories(self, trajectories: dict):
        '''Flatten the collected trajectories.

        `collect_trajectories` returns a dict with, for each agent, N collected trajectories. 
        This function flatten the N trajectories into one "big" trajectory, such that we can then sample minbatches.
        '''
        pass

    @abc.abstractmethod
    def add_gae(self, trajectories, gamma=0.99, lam=0.95):
        '''Add Generalized Advantage Estimations to collected trajectories.'''
        pass

    @abc.abstractmethod
    def update_critic(self, flattened: dict):
        '''Update the critic(s).'''
        pass