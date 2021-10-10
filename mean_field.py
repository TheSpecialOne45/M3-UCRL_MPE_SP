import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from kornia.enhance import histogram2d

class MeanField():
    def __init__(self, n_agents, cell_size=0.6, bandwidth=0.1, density=True):
        world_size = 0.5*np.sqrt(n_agents)
        self.max = world_size + 2*cell_size
        self.min = - self.max + cell_size
        self.cell_size = cell_size
        self.edges = torch.arange(self.min, self.max, cell_size)
        self.center_bins = self.edges[:-1] + cell_size/2
        self.x = torch.zeros(n_agents)
        self.y = torch.zeros(n_agents)
        self.density = density
        self.bandwidth = torch.tensor(bandwidth)

    def update(self, states):
        if not isinstance(states, torch.Tensor): 
            states = torch.from_numpy(np.asarray(states))
        self.x = states[:,2].unsqueeze(0)
        self.y = states[:,3].unsqueeze(0)
        self.dist = histogram2d(self.x, self.y, self.center_bins, self.bandwidth)
        self.dist = self.dist.flatten()
        return self.dist

    def get_dist(self):
        return torch.from_numpy(self.dist.flatten())
         
    def plot(self):
        H = np.asarray(self.dist.clone().detach().cpu()).T
        fig, ax = plt.subplots()
        ax.plot(title='mean-field positions')
        plt.imshow(H, interpolation='nearest', origin='lower', extent=[self.edges[0], self.edges[-1], self.edges[0], self.edges[-1]])
        plt.show()
