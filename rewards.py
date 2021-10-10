from networks import NNModel
import torch
from utils import preprocess
import numpy as np



def reward_function(device, states, targets, actions, mean_field_dist, collision_coeff, target_coeff, action_coeff):
    n_agents = states.shape[0]
    rewards = torch.zeros(n_agents).to(device)
    n_collisions = 0
    n_zero_action = 0
    n_target_reached = 0

    for i, state_current in enumerate(states):
        for j, state_other in enumerate(states):
            if i != j:
                r_col = soft_is_collision(state_current[2:], state_other[2:], collision_coeff)
                if collision_coeff == 0:
                    n_collisions += r_col.item()/1
                else:
                    n_collisions += r_col.item()/collision_coeff
                rewards[i] += r_col
                
        rewards[i] += - torch.sum((state_current[2:]- targets[i])**2)

        r_target = soft_is_on_target(state_current[2:], targets[i], target_coeff)
        rewards[i] += r_target
        if target_coeff == 0:
            n_target_reached += r_target.item()/1
        else:
            n_target_reached += r_target.item()/target_coeff

        r_action = soft_action_penalty(state_current[:2], action_coeff)
        rewards[i] += r_action
        if action_coeff == 0:
            n_zero_action += r_action.item()/1
        else:
            n_zero_action += r_action.item()/action_coeff
    
    n_collisions /= 2
    completion_rate = n_target_reached / n_agents
    collision_rate = n_zero_action / n_agents
    zero_action_rate = n_zero_action / n_agents
    return rewards, completion_rate, collision_rate, zero_action_rate


def soft_is_collision(pos1, pos2, collision_coeff=1):
    dist_min = 0.3
    delta_pos = pos1 - pos2
    dist = torch.sqrt(torch.sum(torch.square(delta_pos)))
    penalty = -0.5*collision_coeff*(1+torch.tanh(-100*(dist-dist_min)))
    return penalty

def soft_is_on_target(pos1, pos2, target_coeff=1):
    dist_min = 0.1
    delta_pos = pos1 - pos2
    dist = torch.sqrt(torch.sum(torch.square(delta_pos)))
    arrived = 0.5*target_coeff*(1+torch.tanh(-100*(dist-dist_min)))
    return arrived

def soft_action_penalty(state_vel, action, action_coeff=1):
    penalty = - action_coeff*torch.sum(torch.abs(state_vel))
    return penalty
