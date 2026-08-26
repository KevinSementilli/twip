from isaaclab.envs import mdp

def set_curr_param(env, env_ids, old_value, value, start_step): 

    if env.common_step_counter >= start_step: 
        return value 

    return mdp.modify_term_cfg.NO_CHANGE