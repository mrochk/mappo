from mappo import IPPO
from env import create_environment
from stable_baselines3 import PPO

env = create_environment(2, 2500)
algo = IPPO(env, batch_size=256, c_ent=0.01)

#loaded = PPO.load('pretrained.zip')

#for critic in algo.critics:
    #algo.critics[critic].value_net_in.load_state_dict(loaded.policy.mlp_extractor.value_net.state_dict())
    #algo.critics[critic].value_net_out.load_state_dict(loaded.policy.value_net.state_dict())

#for actor in algo.actors:
    #algo.actors[actor].policy_net.load_state_dict(loaded.policy.mlp_extractor.policy_net.state_dict())
    #algo.actors[actor].action_net.load_state_dict(loaded.policy.action_net.state_dict())

#del loaded

print(algo)

algo.learn(niters=100, nsteps=8192, checkpoint_path='ippo_nopretrain', eval_path='ippo_nopretrain')
