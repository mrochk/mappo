from mappo import MAPPO
from env import create_environment

if __name__ == '__main__':

    env = create_environment(2, 2500)
    algo = MAPPO(env, batch_size=256, c_ent=0.01)

    ### uncomment to load pretrained model:

    '''
    from stable_baselines3 import PPO
    loaded = PPO.load('pretrained.zip')
    for critic in algo.critics:
        algo.critics[critic].value_net_in.load_state_dict(loaded.policy.mlp_extractor.value_net.state_dict())
        algo.critics[critic].value_net_out.load_state_dict(loaded.policy.value_net.state_dict())
    for actor in algo.actors:
        algo.actors[actor].policy_net.load_state_dict(loaded.policy.mlp_extractor.policy_net.state_dict())
        algo.actors[actor].action_net.load_state_dict(loaded.policy.action_net.state_dict())
    del loaded
    '''

    print(algo)
    algo.learn(
        niters=10, 
        nsteps=8192,
    )

