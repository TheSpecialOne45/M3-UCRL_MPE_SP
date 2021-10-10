from networks import NNModel
import torch
from utils import preprocess
import numpy as np
from scipy.special import softmax
import random

class Policy():
    def __init__(self, model_cfg, network_cfg, optimizer_cfg, loss_cfg=None, seed=0, scheduler_cfg=None):
        self.model = NNModel(network_cfg, optimizer_cfg, loss_cfg, seed, scheduler_cfg)
        self.model.to(model_cfg["device"])
        self.dim_action = model_cfg["dim_action"]
        self.dim_state = model_cfg["dim_state"]
        self.dim_target = model_cfg["dim_target"]

    def act(self, states, targets, mean_fields, tau, eps=0.):
        action_dist = None
        if random.random() > eps:
            x = torch.cat([states, targets, mean_fields], dim=1)
            action_dist = self.model(x, tau).squeeze(0)
        else:
            action = np.random.randint(5)
            action_dist = torch.nn.functional.one_hot(torch.tensor(action), self.dim_action)
        return action_dist
