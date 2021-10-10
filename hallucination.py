from networks import NNModel
import torch
from utils import preprocess
import numpy as np

class Hallucination():
    def __init__(self, model_cfg, network_cfg, optimizer_cfg, loss_cfg=None, seed=0, scheduler_cfg=None):
        self.model = NNModel(network_cfg, optimizer_cfg, loss_cfg, seed, scheduler_cfg)
        self.model.to(model_cfg["device"])
        self.dim_state = model_cfg["dim_state"]

    def hallucinate(self, states, targets, mean_fields):
        x = torch.cat([states, targets, mean_fields], dim=1)
        etas = self.model(x)
        return etas
        