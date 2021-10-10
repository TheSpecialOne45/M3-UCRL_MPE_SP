from torch import from_numpy
import numpy as np
import torch
from timeit import default_timer
from PIL import Image

NUM_ACTIONS = 5
MAX_SPEED = 1

def args_to_cfg(args, dim_action, dim_state, dim_target, dim_mean_field):
    device = None
    if args["use_gpu"]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    env, dyn, rew, pol, hal = dict(), dict(), dict(), dict(), dict()

    for k, v in args.items():
        if "env" in k:
            env[k.replace("env_", "")] = v
        elif "dyn" in k:
            dyn[k.replace("dyn_", "")] = v
        elif "pol" in k:
            pol[k.replace("pol_", "")] = v
        elif "hal" in k:
            hal[k.replace("hal_", "")] = v

    map_size = np.sqrt(args["env_N"])
    seed = args["env_seed"]

    env_cfg = env
    dyn_cfg = {
        "model_cfg": {"dim_action": dim_action, "dim_state": dim_state, "dim_mean_field": dim_mean_field, "n_networks": dyn["n_networks"], "device": device},
        "network_cfg": {
            "type": "NN",
            "parameters": {
                "device": device,
                "input_size": dim_state + dim_action + dim_mean_field,
                "output_size": 2*dim_state,
                "layers": make_layers(dyn),
                "non_linearity": dyn["non_linearity"],
                "normalize_cfg": {"x_min": [-MAX_SPEED, -map_size, 0, 0], "x_max": [MAX_SPEED, map_size, 1, 1], "l": [-1, -1, 0, 0], "u": [1, 1, 1, 1], "lengths": [int(dim_state/2), int(dim_state/2), dim_action, dim_mean_field]}
            },
        },
        "optimizer_cfg": {
            "type": dyn["optimizer"],
            "parameters": {
                "lr": dyn["lr"]
            }
        },
        "loss_cfg": {
            "type": dyn["loss"],
            "parameters": {}
        },
        "seed":seed,
        "scheduler_cfg":None
    }


    pol_cfg = {
        "model_cfg": {"dim_action": dim_action, "dim_state": dim_state, "dim_target": dim_target, "device": device},
        "network_cfg": {
            "type": "NN",
            "parameters": {
                "device": device,
                "input_size": dim_state + dim_target + dim_mean_field,
                "output_size": dim_action,
                "layers": make_layers(pol),
                "non_linearity": pol["non_linearity"],
                "final_activation": pol["final_activation"],
                "normalize_cfg": {"x_min": [-MAX_SPEED, -map_size, 0], "x_max": [MAX_SPEED, map_size, 1], "l": [-1, -1, 0], "u": [1, 1, 1], "lengths": [int(dim_state/2), dim_state, dim_mean_field]}
            },
        },
        "optimizer_cfg": {
            "type": pol["optimizer"],
            "parameters": {
                "lr": pol["lr"],
            }
        },
        # "scheduler_cfg": {
        #     "type": pol["scheduler"],
        #     "parameters": {
        #         "T_0": pol["T_0"]
        #     }
        # },

        "loss_cfg": {
            "type": "None",
            "parameters": {}
        },
        "seed":seed,
        "scheduler_cfg":None

    }

    hal_cfg = {
        "model_cfg": {"dim_action": dim_action, "dim_state": dim_state, "dim_target": dim_target, "device": device},
        "network_cfg": {
            "type": "NN",
            "parameters": {
                "device": device,
                "input_size": dim_state + dim_target + dim_mean_field,
                "output_size": dim_state,
                "layers": make_layers(hal),
                "non_linearity": hal["non_linearity"],
                "final_activation": hal["final_activation"],
                "normalize_cfg": {"x_min": [-MAX_SPEED, -map_size, 0], "x_max": [MAX_SPEED, map_size, 1], "l": [-1, -1, 0], "u": [1, 1, 1], "lengths": [int(dim_state/2), dim_state, dim_mean_field]}

            },
        },
        "optimizer_cfg": {
            "type": hal["optimizer"],
            "parameters": {
                "lr": hal["lr"]
            }
        },
        "loss_cfg": {
            "type": "None",
            "parameters": {}
        },
        "seed":seed,
        "scheduler_cfg":None

    }
    return device, env_cfg, dyn_cfg, pol_cfg, hal_cfg

    

def make_layers(dict):
    layers = (dict["width"],) * dict["n_layers"]
    return layers

def make_ensemble_losses_dict(losses, n_networks):
    losses_dict = {}
    for i, loss in enumerate(losses[:-1]):
        losses_dict[f"nn_{i}"] = loss
    losses_dict["avg_loss"] = losses[-1]
    return losses_dict

def make_uncertainties_dict(uncertainties, dim_state):
    uncertainties_dict = {}
    for i in range(dim_state):
        uncertainties_dict[f"state_comp_{i}"] = torch.mean(uncertainties[:,i])
    return uncertainties_dict

class Timer(object):
    """
    Utility to measure times.

    """

    def __init__(self):
        self.total_time = 0.0
        self.start_time = 0.0
        self.end_time = 0.0

    def start(self):
        self.start_time = default_timer()

    def end(self):
        self.total_time += default_timer() - self.start_time

    def get(self):
        return self.total_time

    def get_current(self):
        return default_timer() - self.start_time

    def reset(self):
        self.__init__()

    def __repr__(self):
        return self.get()


def format_action_prob(actions):
    action_probs = np.zeros(NUM_ACTIONS)
    for i in range(NUM_ACTIONS):
        action_probs[i] = actions.count(i)
    action_probs = action_probs / action_probs.sum()
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "→", "↓", "↑"]

    # if np.amax(action_probs) == 1.0:
    #     print("Action probs == 1, terminating run ...")
    #     quit()
        
    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.2f}".format(action_prob) + " "

    return buffer

def generate_gif(device, path, env, policy, mean_field, tau, max_cycles, eps=0, fps=30):
    imgs = []
    obs, targets = env.reset()

    dim_mean_field = (len(mean_field.edges)-1)**2
    dim_action = 5 
    dim_state = 4
    dim_target = 2
    n_agents = len(env.agents)
    
    targets = torch.FloatTensor(list(targets.values())).to(device)
    states = torch.zeros((max_cycles, n_agents, dim_state)).to(device)
    actions_dist = torch.zeros((max_cycles, n_agents, dim_action)).to(device)
    mean_fields = torch.zeros((max_cycles, dim_mean_field)).to(device)

    for step in range(max_cycles):
        states[step] = torch.FloatTensor(list(obs.values()))
        mean_fields[step] = mean_field.update(list(obs.values()))
        actions_dist[step] = policy.act(states[step], targets, mean_fields[step].repeat(n_agents, 1), tau).detach()
        act = {agent: int(torch.argmax(actions_dist[step, i]).item()) for i, agent in enumerate(env.agents)}
        obs, _, _, _ = env.step(act)

        ndarray = env.render(mode='rgb_array')
        im = Image.fromarray(ndarray).convert('P')
        imgs.append(im)

    env.close()
    hundredths = max(int(fps/100), 4)
    img, *imgs = imgs
    img.save(fp=path, format='GIF', append_images=imgs,
         save_all=True, duration=hundredths, loop=0, include_color_table=True, optimize=True)