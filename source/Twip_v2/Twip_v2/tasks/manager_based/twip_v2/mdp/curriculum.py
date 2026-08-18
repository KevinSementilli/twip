from isaaclab.envs import mdp


def set_disturbance(env, env_ids, old_value, x_range,y_range, start_step):

    if env.common_step_counter >= start_step:
        return {
            "x": x_range,
            "y": y_range,
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

    return mdp.modify_term_cfg.NO_CHANGE


def set_curr_param(env, env_ids, old_value, value, start_step): 

    if env.common_step_counter >= start_step: 
        return value 

    return mdp.modify_term_cfg.NO_CHANGE