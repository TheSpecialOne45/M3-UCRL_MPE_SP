import random
from pettingzoo.mpe import simple_spread_v2
import numpy as np
import torch
import time 
from statistics import mean
from argparse import ArgumentParser
from utils import args_to_cfg, make_ensemble_losses_dict, make_uncertainties_dict, Timer, format_action_prob, generate_gif, NUM_ACTIONS
from dynamics_model import DynamicsModel
from rewards import reward_function
from policy import Policy, RandomPolicy, BaselinePolicy, BaselinePolicy2
from hallucination import Hallucination
from mean_field import MeanField
from torch.utils.tensorboard import SummaryWriter
import datetime
try:
    import wandb
except ImportError:
    print("Install wandb to log to Weights & Biases")
import warnings
import pickle

warnings.simplefilter("ignore", DeprecationWarning)
torch.autograd.set_detect_anomaly(True)

##########################################################################
######################### Parsing Parameters #############################
##########################################################################

parser = ArgumentParser()
parser.add_argument("--n_episodes", help="number of episodes to run", default=1000, type=int)
parser.add_argument("--interval_print", help="interval print updates", default=1, type=int)
parser.add_argument("--interval_render", help="interval rendering", default=25, type=int)
parser.add_argument("--interval_checkpoint", help="interval saving checkpoint", default=500, type=int)
parser.add_argument("--interval_gif", help="interval saving gif", default=100, type=int)
parser.add_argument("--render", help="render episode", default=False, type=bool) 
parser.add_argument("--save_gif", help="save episode gif", default=True, type=bool)
parser.add_argument("--save_checkpoint", help="save training checkpoint", default=False, type=bool)
parser.add_argument("--load_checkpoint", help="load models checkpoint", default=False, type=bool)
parser.add_argument("--use_gpu", help="use GPU if available", default=False, type=bool)
parser.add_argument("--beta", help="hallucination beta", default=1., type=float)
parser.add_argument("--eps_start", help="max exploration", default=0.95, type=float)
parser.add_argument("--eps_end", help="min exploration", default=0.05, type=float)
parser.add_argument("--eps_decay", help="exploration decay", default=0.99, type=float)
parser.add_argument("--cell_size", help="mean-field position bin", default=1.5, type=float)
parser.add_argument("--collision_coeff", help="penalty coeff for collision", default=1., type=float)
parser.add_argument("--target_coeff", help="reward coeff for reaching target", default=1., type=float)
parser.add_argument("--action_coeff", help="penalty coeff for moving", default=0., type=float)
parser.add_argument("--env_N", help="number of agents", default=10, type=int)
parser.add_argument("--env_local_ratio", help="local/global reward ratio", default=1., type=float)
parser.add_argument("--env_max_cycles", help="maximum number of steps per episode", default=30, type=int)
parser.add_argument("--env_seed", help="random seed index", default=0, type=int)

parser.add_argument("--dyn_n_networks", help="number of networks in the ensemble for the dynamics model", default=5, type=int)
parser.add_argument("--dyn_n_layers", help="number of layers for the dynamics model - 1", default=2, type=int)
parser.add_argument("--dyn_width", help="width of layers for the dynamics model", default=64, type=int)
parser.add_argument("--dyn_non_linearity", help="activation function type for the dynamics model", default="Swish", type=str)
parser.add_argument("--dyn_optimizer", help="optimizer for the dynamics model", default="Adam", type=str)
parser.add_argument("--dyn_lr", help="learning rate for the dynamics model", default=1e-3, type=float)
parser.add_argument("--dyn_loss", help="loss for the dynamics model", default="Gaussian_NLL", type=str)

parser.add_argument("--pol_n_layers", help="number of layers for the policy model - 1", default=4, type=int)
parser.add_argument("--pol_width", help="width of layers for the policy model", default=128, type=int)
parser.add_argument("--pol_non_linearity", help="activation function type for the policy model", default="Swish", type=str)
parser.add_argument("--pol_final_activation", help="activation function type for the final layer of the policy model", default=None, type=str)
parser.add_argument("--pol_optimizer", help="optimizer for the policy model", default="Adam", type=str)
parser.add_argument("--pol_lr", help="learning rate for the policy model", default=1e-3, type=float)
parser.add_argument("--pol_gumbel_tau", help="tau value of the gumble softmax layer for the policy model", default=10., type=float)

parser.add_argument("--hal_n_layers", help="number of layers for the hallucination model", default=2, type=int)
parser.add_argument("--hal_width", help="width of layers for the hallucination model", default=64, type=int)
parser.add_argument("--hal_non_linearity", help="activation function type for the hallucination model", default="Swish", type=str)
parser.add_argument("--hal_final_activation", help="activation function type for the final layer of the hallucination model", default="Tanh", type=str)
parser.add_argument("--hal_optimizer", help="optimizer for the hallucination model", default="Adam", type=str)
parser.add_argument("--hal_lr", help="learning rate for the hallucination model", default=1e-3, type=float)

args = parser.parse_args()
#wandb.init(sync_tensorboard=True, config=args)
wandb.init(config=args)
config = wandb.config
args = dict(config)
#args = vars(args)

seed_index=args["env_seed"]
n_episodes = args["n_episodes"]
eps_start = args["eps_start"]
eps_end = args["eps_end"]
eps_decay = args["eps_decay"]
cell_size = args["cell_size"]
tau = args["pol_gumbel_tau"]
n_agents = args["env_N"]
max_cycles = args["env_max_cycles"]
render = args["render"]
load_checkpoint = args["load_checkpoint"]
beta = args["beta"]
collision_coeff=args["collision_coeff"]
target_coeff=args["target_coeff"]
action_coeff=args["action_coeff"]
render_interval = args["interval_render"]
print_interval = args["interval_print"]
save_checkpoint = args["save_checkpoint"]
checkpoint_interval = args["interval_checkpoint"]
save_gif = args["save_gif"]
gif_interval = args["interval_gif"]

# random seed selection
random.seed(0)
seeds = random.sample(range(100000),10)
seed = seeds[seed_index]
args["env_seed"]=seed
print("SEEDS : ", seeds, ", using seed ", seed)

# logging to wandb and tensorboard
log_dir = "experiments_runs/ " + "comparison_baseline_mf_PROB/{}".format(seed_index)  #+ datetime.datetime.now().strftime("%m-%d-%Hh%Mm%Ss")
wandb.tensorboard.patch(root_logdir=log_dir, pytorch=True)
writer = SummaryWriter(log_dir=log_dir)
writer.add_hparams(args, {})

# Initializing the models and the environment
mean_field = MeanField(n_agents, cell_size)
dim_mean_field = (len(mean_field.edges)-1)**2
dim_action = 5 
dim_state = 4
dim_target = 2

device, env_cfg, dyn_cfg, pol_cfg, hal_cfg = args_to_cfg(args, dim_action, dim_state, dim_target, dim_mean_field)
env = simple_spread_v2.parallel_env(**env_cfg)
dynamics_model = DynamicsModel(**dyn_cfg)
policy = Policy(**pol_cfg)
hallucination = Hallucination(**hal_cfg)


if load_checkpoint:
    print("Loading checkpoint ...")
    checkpoint_path = "path_of_checkpoint"
    checkpoint = torch.load(checkpoint_path)
    dynamics_model.model.load_state_dict(checkpoint["dynamics_state_dict"])
    policy.model.load_state_dict(checkpoint["policy_state_dict"])
    hallucination.model.load_state_dict(checkpoint["hallucination_state_dict"])


total_timer = Timer()
episode_timer = Timer()

##################################################################
##################### Training ###################################
##################################################################

total_timer.start()
for episode in range(n_episodes):

    # Rollout in the real environment to update dynamics model

    episode_timer.start()
    obs, targets = env.reset()
    targets = torch.FloatTensor(list(targets.values())).to(device)
    states = torch.zeros((max_cycles, n_agents, dim_state)).to(device)
    next_states = torch.zeros((max_cycles, n_agents, dim_state)).to(device)
    actions_dist = torch.zeros((max_cycles, n_agents, dim_action)).to(device)
    mean_fields = torch.zeros((max_cycles, n_agents, dim_mean_field)).to(device)
    rewards = []
    actions_env = []
    completion_rates = []
    collision_rates = []
    zero_action_rates = []

    for step in range(max_cycles):
        states[step] = torch.FloatTensor(list(obs.values()))
        mean_fields[step] = mean_field.update(list(obs.values())).repeat(n_agents, 1)
        actions_dist[step] = policy.act(states[step], targets, mean_fields[step], tau, eps_start).detach()
        act = {agent: int(torch.argmax(actions_dist[step, i]).item()) for i, agent in enumerate(env.agents)}
        obs, rew, _, _ = env.step(act)
        next_states[step] = torch.FloatTensor(list(obs.values()))
        actions_env.extend(act.values())
        rewards.extend(rew.values())
        completion_rate, collision_rate, zero_action_rate = env.aec_env.env.env.scenario.assess_world(env.aec_env.env.env.world)
        completion_rates.append(completion_rate)
        collision_rates.append(collision_rate)
        zero_action_rates.append(zero_action_rate)
        if render and episode % render_interval == 0:
            env.render()
            time.sleep(0.05)
    
    if save_gif and episode % gif_interval == 0:
        generate_gif(device, log_dir+f"/episode_{episode}.gif", env, policy, mean_field, tau, max_cycles)

    cumulative_reward = sum(rewards)/(n_agents*max_cycles)
    dyn_losses = dynamics_model.update_model(states, actions_dist, next_states, mean_fields)

    if episode % print_interval == 0:
        print(
            '\r Episode {}'
            '\t 🏆 Normalized Cumulative Reward: {:.2f}'
            '\t ✅ Completion: {:.2f}'
            '\t 💥 Collision: {:.3f}'
            '\t 🔀 Action Probs: {}'
            .format(
                episode,
                cumulative_reward,
                mean(completion_rates), 
                mean(collision_rates),
                format_action_prob(actions_env)
            ))
    



    # Simulation to update the policy 

    states = states[0].clone().detach() # initial states
    mean_field_dist = mean_field.update(states).repeat(n_agents, 1).to(device)
    loss_cumulative_reward = 0
    policy.model.optimizer.zero_grad()
    hallucination.model.optimizer.zero_grad()
    aleatoric_uncertainties = torch.zeros(max_cycles, dim_state)
    epistemic_uncertainties = torch.zeros(max_cycles, dim_state)
    hallucinations = []
    actions_sim = []
    sim_completion_rates = []
    sim_collision_rates = []
    sim_zero_action_rates = []

    for step in range(max_cycles):
        act = policy.act(states, targets, mean_field_dist, tau)
        actions_sim.extend(torch.argmax(act, dim=1).tolist())
        etas = hallucination.hallucinate(states, targets, mean_field_dist) 
        next_states_mean, next_states_var = dynamics_model.simulate(states, act, mean_field_dist)
        states = next_states_mean + beta * etas * next_states_var
        hallucinations.append(torch.max(torch.abs(beta*etas*next_states_var)).item())
        aleatoric, epistemic = dynamics_model.get_average_uncertainties()
        aleatoric_uncertainties[step] = aleatoric
        epistemic_uncertainties[step] = epistemic
        mean_field_dist = mean_field.update(states).repeat(n_agents, 1).to(device)
        rews, completion_rate, collision_rate, zero_action_rate = reward_function(device, states, targets, act, mean_field_dist, collision_coeff, target_coeff, action_coeff)
        sim_completion_rates.append(completion_rate)
        sim_collision_rates.append(collision_rate)
        sim_zero_action_rates.append(zero_action_rate)
        loss_cumulative_reward -= torch.sum(rews)

    loss_cumulative_reward /= n_agents*max_cycles
    loss_cumulative_reward.backward()
    torch.nn.utils.clip_grad_norm_(policy.model.layers.parameters(), max_norm=1, norm_type='inf')
    torch.nn.utils.clip_grad_norm_(hallucination.model.layers.parameters(), max_norm=1, norm_type='inf')
    for model in dynamics_model.model.ensemble:
        torch.nn.utils.clip_grad_norm_(model.layers.parameters(), max_norm=1, norm_type='inf')
    policy.model.optimizer.step()
    hallucination.model.optimizer.step()
    episode_timer.end()

  
    # Logging to tensorboard

    writer.add_scalar("sim_cumulative_reward", -loss_cumulative_reward, episode)
    writer.add_scalars("aleatoric_uncertainties", make_uncertainties_dict(aleatoric_uncertainties, dim_state), episode)
    writer.add_scalars("epistemic_uncertainties", make_uncertainties_dict(epistemic_uncertainties, dim_state), episode)
    writer.add_scalar("max_hal", max(hallucinations),  episode)
    writer.add_histogram("sim_actions", np.array(actions_sim), episode)
    writer.add_scalar("sim_completion_rates", mean(sim_completion_rates[int(max_cycles/2):]), episode)
    writer.add_scalar("sim_collision_rates", mean(sim_collision_rates), episode)
    writer.add_scalar("sim_zero_action_rates", mean(sim_zero_action_rates[int(max_cycles/2):]), episode)
    writer.add_scalar("env_cumulative_reward", cumulative_reward, episode)
    writer.add_scalars("dynamics_loss", make_ensemble_losses_dict(dyn_losses, dyn_cfg["model_cfg"]["n_networks"]), episode)
    writer.add_histogram("env_actions", np.array(actions_env), episode)
    writer.add_scalar("env_completion_rates", mean(completion_rates[int(max_cycles/2):]), episode)
    writer.add_scalar("env_collision_rates", mean(collision_rates), episode)
    writer.add_scalar("env_zero_action_rates", mean(zero_action_rates[int(max_cycles/2):]), episode)
    writer.add_histogram("reward_distributions", np.array(rewards), episode, bins="sqrt")
    writer.add_scalar("env_eps", eps_start, episode)
    writer.add_scalar("time_episode", episode_timer.get(), episode)

    episode_timer.reset()

    # Epsilon decay
    eps_start = max(eps_end, eps_decay * eps_start)
  

    if save_checkpoint and episode % checkpoint_interval == 0: 
        print("Saving checkpoint ...")
        torch.save({
            "episode": episode,
            "dynamics_state_dict": dynamics_model.model.state_dict(),
            "policy_state_dict": policy.model.state_dict(), 
            "hallucination_state_dict": hallucination.model.state_dict(),
        }, log_dir + f"/checkpoint_{episode}")

if save_checkpoint:
    print("Saving checkpoint ...")
    torch.save({
        "episode": episode,
        "dynamics_state_dict": dynamics_model.model.state_dict(),
        "policy_state_dict": policy.model.state_dict(), 
        "hallucination_state_dict": hallucination.model.state_dict(),
    }, log_dir + "/checkpoint_final")

total_timer.end()
print("Total training time : ", total_timer.get())
if save_gif:    
    generate_gif(device, log_dir+f"/episode_final.gif", env, policy, mean_field, tau, max_cycles)