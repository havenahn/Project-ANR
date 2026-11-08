from envs.unicycle_env import UnicycleEnv

"""
This file includes a function that simply returns one of the supported environments. 
"""

def build_env(env_name, obs_config='default', rand_init=False, init_mode='manual', init_grid_n=10, init_x_min=-2.0, init_x_max=2.0, init_y_min=-2.0,init_y_max=2.0,
    init_theta_min=-2.0, init_theta_max=2.0,init_v_min=-2.0, init_v_max=2.0,ref_x=0.0,ref_y=0.0, ref_theta=0.0, ref_v=0.0, manual_init_state=None,
    custom_obstacles=None, r_a=1.0, r_w=1.0,):

    """Build the Unicycle environment."""
    if env_name != 'Unicycle':
        raise Exception("Only env_name='Unicycle' is supported in this trimmed project.")
    return UnicycleEnv(obs_config, rand_init=rand_init, init_mode=init_mode,init_grid_n=init_grid_n, init_x_min=init_x_min,
        init_x_max=init_x_max, init_y_min=init_y_min,init_y_max=init_y_max, init_theta_min=init_theta_min, init_theta_max=init_theta_max,
        init_v_min=init_v_min,init_v_max=init_v_max, ref_x=ref_x,ref_y=ref_y, ref_theta=ref_theta, ref_v=ref_v, manual_init_state=manual_init_state,
        custom_obstacles=custom_obstacles, r_a=r_a,r_w=r_w,)
