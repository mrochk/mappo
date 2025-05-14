import numpy as np
import torch as th
import sys

from env import create_environment 
from mappo import MAPPO, IPPO

assert len(sys.argv) > 1

env = create_environment(2, 2500, render_mode='human')
filename = sys.argv[1]

if 'ippo' in filename: algo = IPPO(env)
else: algo = MAPPO(env)

algo.load_state_dict(th.load(filename))
algo = algo.eval()

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