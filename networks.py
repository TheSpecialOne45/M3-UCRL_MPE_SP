import torch
import torch.nn as nn
import numpy as np

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)

class DetEnsembleModel(nn.Module):
    def __init__(
        self, n_networks, network_cfg, optimizer_cfg, loss_cfg, seed, scheduler_config=None
    ):
        super().__init__()
        seeds = np.random.randint(1e4, size=n_networks)
        self.ensemble = [
            NNModel(network_cfg, optimizer_cfg, loss_cfg, seeds[i], scheduler_config)
            for i in range(n_networks)
        ]
        self.n_networks = n_networks
        self.network_cfg = network_cfg

    def forward(self, x):
        means = torch.zeros(
            x.shape[0], self.n_networks, self.network_cfg["parameters"]["output_size"]
        ).to(self.network_cfg["parameters"]["device"])
    
        for i, model in enumerate(self.ensemble):
            means[:, i, :] = model(x)

        mean = torch.mean(means, axis=1)
        var = torch.mean((means - mean.unsqueeze(1)) ** 2, axis=1)
        return mean, var

    def fit(self, inputs, true_outputs):
        losses = []
        for model in self.ensemble:
            losses.append(model.fit(inputs, true_outputs))
        losses.append(sum(losses) / len(losses))
        return losses


class ProbEnsembleModel(nn.Module):
    def __init__(
        self, n_networks, network_cfg, optimizer_cfg, loss_cfg, seed, scheduler_config=None
    ):
        super().__init__()
        seeds = np.random.randint(1e4, size=n_networks)
        self.ensemble = [
            PNNModel(network_cfg, optimizer_cfg, loss_cfg, seeds[i], scheduler_config)
            for i in range(n_networks)
        ]
        self.n_networks = n_networks
        self.network_cfg = network_cfg

    def forward(self, x):
        means = torch.zeros(x.shape[0],
            self.n_networks, self.network_cfg["parameters"]["output_size"] // 2
        )
        vars = torch.zeros(x.shape[0],
            self.n_networks, self.network_cfg["parameters"]["output_size"] // 2
        )

        for i, model in enumerate(self.ensemble):
            means[:, i, :], vars[:, i, :] = model(x)

        mean = torch.mean(means, axis=1)
        self.aleatoric = torch.mean(vars, axis=1)
        self.epistemic = torch.mean((means - mean.unsqueeze(1)) ** 2, axis=1)
        var = self.aleatoric + self.epistemic
        return mean, var

    def get_average_uncertainties(self):
        aleatoric_means = torch.mean(self.aleatoric, axis=0)
        epistemic_means = torch.mean(self.epistemic, axis=0)
        return aleatoric_means, epistemic_means

    def fit(self, inputs, true_outputs):
        losses = []
        for model in self.ensemble:
            losses.append(model.fit(inputs, true_outputs))
        losses.append(sum(losses) / len(losses))
        return losses


class NNModel(nn.Module):
    def __init__(
        self, network_cfg, optimizer_cfg, loss_cfg, seed=0, scheduler_cfg=None
    ):
        super().__init__()
        torch.manual_seed(seed)
        self._build_network(**network_cfg["parameters"])
        self.optimizer = self._parse_optimizer(optimizer_cfg)
        self.scheduler = self._parse_scheduler(scheduler_cfg)
        self.loss = self._parse_loss(loss_cfg)
        self.losses = []
        self.apply(init_weights)
        

    def forward(self, x, tau=None):
        y = self.layers(x)
        if tau is not None:
            y = torch.nn.functional.gumbel_softmax(y, tau)
        return y

    def fit(self, inputs, true_outputs):
        self.train()
        self.optimizer.zero_grad()
        outputs = self(inputs)
        if isinstance(self.loss, torch.nn.GaussianNLLLoss):
            means = outputs[:, :int(outputs.shape[1]/2)]
            var = self._softplus(outputs[:, int(outputs.shape[1]/2):])
            loss = self.loss(means, true_outputs, var)
        else:
            loss = self.loss(outputs, true_outputs)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.layers.parameters(), max_norm=1, norm_type='inf')

        if self.scheduler is not None:
            self.scheduler.step()
        else:
            self.optimizer.step()
        return loss.item()

    def _build_network(
        self,
        device,
        normalize_cfg,
        input_size,
        output_size,
        layers,
        non_linearity,
        final_activation=None
    ):
        non_linearity_layer = self._parse_nonlinearity(non_linearity)

        self.input_size = input_size
        self.output_size = output_size
        self.layers = []
        in_features = input_size

        self.layers.append(NormalizeLayer(normalize_cfg, device))
        for layer in layers:
            self.layers.append(nn.Linear(in_features, layer))
            self.layers.append(non_linearity_layer)
            in_features = layer

        self.layers.append(nn.Linear(in_features, output_size))
        if final_activation is not None:
            final_activation_layer = self._parse_nonlinearity(final_activation)
            self.layers.append(final_activation_layer)

        self.layers = nn.Sequential(*self.layers)

    def _parse_nonlinearity(self, non_linearity):
        """Parse non-linearity."""
        name = non_linearity.lower()
        if name == "relu":
            return nn.ReLU()
        elif name == "leakyrelu":
            return nn.LeakyReLU()
        elif name == "swish":
            return nn.SiLU()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "softmax":
            return nn.Softmax(dim=1)
        else:
            raise NotImplementedError(f"non-linearity {non_linearity} not implemented")

    def _parse_optimizer(self, optimizer_cfg):
        opt_type = optimizer_cfg["type"].lower()
        opt_parameters = optimizer_cfg["parameters"]
        if opt_type == "adam":
            return torch.optim.Adam(self.layers.parameters(), **opt_parameters)
        if opt_type == "sgd":
            return torch.optim.SGD(self.layers.parameters(), **opt_parameters)
        else:
            raise NotImplementedError(f"opt_type {opt_type} not implemented")

    def _parse_loss(self, loss_cfg):
        loss_type = loss_cfg["type"].lower()
        loss_parameters = loss_cfg["parameters"]
        if loss_type == "mse":
            return torch.nn.MSELoss(**loss_parameters)
        elif loss_type == "mae":
            return torch.nn.L1Loss(**loss_parameters)
        elif loss_type == "gaussian_nll":
            return torch.nn.GaussianNLLLoss(**loss_parameters)
        elif loss_type == "none":
            return None
        else:
            raise NotImplementedError(f"loss_type {loss_type} not implemented")

    def _parse_scheduler(self, scheduler_cfg):
        if scheduler_cfg is not None:
            scheduler_type = scheduler_cfg["type"].lower()
            scheduler_parameters = scheduler_cfg["parameters"]
            if scheduler_type == "step":
                return torch.optim.lr_scheduler.StepLR(
                    self.optimizer, **scheduler_parameters
                )
            elif scheduler_type == "cosine":
                return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, **scheduler_parameters
                )
            else:
                raise NotImplementedError(
                    f"scheduler_type {scheduler_type} not implemented"
                )
        return None
    
    def _softplus(self, x):
        """ Compute softplus """
        softplus = torch.log(1+torch.exp(x))
        # Avoid infinities due to taking the exponent
        softplus = torch.where(softplus==float('inf'), x, softplus)
        return softplus


# Inspired by https://github.com/github-jnauta/pytorch-pne/blob/master/models/pnn_2d.py
class PNNModel(NNModel):
    def __init__(
        self, network_cfg, optimizer_cfg, loss_cfg, seed, scheduler_cfg=None
    ):
        super().__init__(network_cfg, optimizer_cfg, loss_cfg, seed, scheduler_cfg)
        assert isinstance(
            self.loss, torch.nn.GaussianNLLLoss
        ), "Probabilistic Model requires GaussianNLLLoss"

    def forward(self, x):
        out = self.layers(x)
        mean, var = torch.split(out, self.output_size // 2, dim=1)
        var = self._softplus(var)
        return mean, var

    def fit(self, inputs, true_outputs):
        self.train()
        self.optimizer.zero_grad()
        outputs_mean, outputs_var = self(inputs)
        loss = self.loss(outputs_mean, true_outputs, outputs_var)
        loss.backward()

        if self.scheduler is not None:
            self.scheduler.step()
        else:
            self.optimizer.step()
        return loss.item()

    def _softplus(self, x):
        """Compute softplus"""
        softplus = torch.log(1 + torch.exp(x))
        # Avoid infinities due to taking the exponent
        softplus = torch.where(softplus == float("inf"), x, softplus)
        return softplus


class NormalizeLayer(nn.Module):
    def __init__(self, normalize_cfg, device):
        super().__init__()
        self.x_max = normalize_cfg["x_max"]
        self.x_min = normalize_cfg["x_min"]
        self.u = normalize_cfg["u"]
        self.l = normalize_cfg["l"]
        self.lengths = normalize_cfg["lengths"]
        self.device = device

    def forward(self, x):
        prev_index = 0
        a = torch.zeros(size=(x.shape[1],)).to(self.device)
        b = torch.zeros(size=(x.shape[1],)).to(self.device)
        prev_index = 0
        for x_min, x_max, u, l, length in zip(self.x_min, self.x_max, self.u, self.l, self.lengths):
            for i in range(prev_index, prev_index+length):
                a[i] = (u - l)/(x_max- x_min)
                b[i] = l - x_min*(u - l)/(x_max - x_min)
            prev_index += length
        x = a*x+b
        x = x.float()
        return x


class SoftArgmaxLayer(nn.Module):
    def __init__(self, beta=1):
        super(SoftArgmaxLayer, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.beta = beta

    def forward(self, x):
        enhanced_values = self.softmax(self.beta * x)
        positions = torch.arange(end=x.size()[1]).float()
        soft_argmax = enhanced_values @ positions
        return soft_argmax


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x)
        return x
