from networks import ProbEnsembleModel, DetEnsembleModel
import torch
from utils import preprocess

class DynamicsModel:

    def __init__(self, model_cfg, network_cfg, optimizer_cfg, loss_cfg, seed=0, scheduler_cfg=None):
        self.model = ProbEnsembleModel(model_cfg["n_networks"], network_cfg, optimizer_cfg, loss_cfg, seed, scheduler_cfg)
        for model in self.model.ensemble:
            model.to(model_cfg["device"])
        self.dim_action = model_cfg["dim_action"]
        self.dim_state = model_cfg["dim_state"]
        self.dim_mean_field = model_cfg["dim_mean_field"]

    def simulate(self, states, actions, mean_fields):
        self.model.eval()
        x = torch.cat([states, actions, mean_fields], dim=1)
        next_states_mean, next_states_covar = self.model(x)
        return next_states_mean, next_states_covar
    
    def update_model(self, states, actions, next_states, mean_fields):
        self.model.train()
        inputs = torch.cat([states, actions, mean_fields], dim=2)
        inputs = inputs.view(-1, inputs.shape[2])
        outputs = next_states
        outputs = outputs.view(-1, outputs.shape[2])
        losses = self.model.fit(inputs, outputs)
        return losses

    def get_average_uncertainties(self):
        aleatoric_means, epistemic_means = self.model.get_average_uncertainties()
        return aleatoric_means, epistemic_means
