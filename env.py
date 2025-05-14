from pettingzoo.butterfly import knights_archers_zombies_v10
import supersuit as ss

def create_environment(num_agents=2, max_cycles=2500, render_mode = None):
    env = knights_archers_zombies_v10.env(
        max_cycles=max_cycles, num_archers=num_agents,
        num_knights=0, max_zombies=2500,
        vector_state=True, render_mode=render_mode)

    return ss.black_death_v3(env)
