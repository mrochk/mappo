import numpy as np
import torch as th

from env import create_environment 

from itertools import count
from mappo import MAPPO

import sys

assert len(sys.argv) > 1

env = create_environment(2, 2500, render_mode='human')
algo = MAPPO(env)
algo.load_state_dict(th.load(sys.argv[1]))
algo.eval()

rewards = []
env.reset()
for agent in env.agent_iter():
    obs, reward, term, trunc, _ = env.last()

    action, _ = algo.actors[agent].action(obs)
    rewards.append(reward)

    if term or trunc: 
        env.close()
        return_ = np.sum(rewards)
        print(return_)
        break

    env.step(action)