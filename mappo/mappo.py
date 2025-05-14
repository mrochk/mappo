import numpy as np
import torch as th
from torch import Tensor
from pettingzoo import AECEnv

from .algorithm import Algorithm
from .critic import CentralizedCritic

class MAPPO(Algorithm):
    '''Multi Agent Proximal Policy Optimization

    Centralized Training - Decentralized Execution Framework: 
    1 Critic for N Actors
    '''

    def __init__(self, env: AECEnv, batch_size: int = 64, epochs: int = 10, eps: float = 0.2, c_ent: float = 0):
        super().__init__(env, batch_size, epochs, eps, c_ent)
        self.centralized_critic = CentralizedCritic(self.ssize, batch_size, 20)

    def flatten_trajectories(self, trajectories: dict):
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

    def collect_trajectories(self, N: int) -> dict:
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

    def split_trajectory(self, trajectory: list) -> dict[str, Tensor]:
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

    def add_gae(self, trajectories: dict, gamma, lam):
        '''Add GAEs using every critic.'''
        return self.centralized_critic.add_gae(trajectories, gamma, lam)

    def update_critic(self, flattened: dict):
        '''Update the centralized critic.'''
        return self.centralized_critic.update(flattened)