from mappo import MAPPO
from env import create_environment
from stable_baselines3 import PPO

env = create_environment(2, 2500)
algo = MAPPO(env, batch_size=256, c_ent=0.01)

#loaded = PPO.load('pretrained.zip')
#algo.centralized_critic.value_net_in.load_state_dict(loaded.policy.mlp_extractor.value_net.state_dict())
#algo.centralized_critic.value_net_out.load_state_dict(loaded.policy.value_net.state_dict())

#for actor in algo.actors:
    #algo.actors[actor].policy_net.load_state_dict(loaded.policy.mlp_extractor.policy_net.state_dict())
    #algo.actors[actor].action_net.load_state_dict(loaded.policy.action_net.state_dict())

##for param in algo.centralized_critic.value_net_in.parameters():
    ##param.requires_grad = False

##for param in algo.centralized_critic.value_net_out.parameters():
    ##param.requires_grad = False

#del loaded

print(algo)

algo.learn(niters=100, nsteps=8192, checkpoint_path='mappo_nopretrain', eval_path='mappo_nopretrain')
