# heat_dataset.py
from typing import Optional
import torch
from torch.utils.data import Dataset
import numpy as np
import networkx as nx
from logic.graphon_generator import sample_graph, interp_graphon
from simulator.heat_simulator import simulate_heat_diffusion


class HeatDiffusionDataset(Dataset):
    """
    On-the-fly PyTorch dataset for heat diffusion on random graphon graphs.
    Each sample is a tensor of shape (T+1, N), the temperature evolution.
    """

    def __init__(
        self,
        n_nodes: int,
        steps: int,
        alpha: float,
        graphon_fn,
        resolution: int = 100,
        transform: Optional[callable] = None
    ):
        """
        Args:
            n_nodes: number of nodes per graph
            steps: number of diffusion timesteps
            alpha: diffusion rate
            graphon_fn: a GraphonFunction (from graphon_generator.interp_graphon)
            resolution: resolution to sample graphon
            transform: optional transform applied to output tensor
        """
        self.n_nodes = n_nodes
        self.steps = steps
        self.alpha = alpha
        self.graphon_fn = graphon_fn
        self.resolution = resolution
        self.transform = transform

    def __len__(self) -> int:
        # infinite / on-the-fly; define an arbitrary large size
        return 10_000

    def __getitem__(self, idx: int) -> torch.Tensor:
        # 1. Sample graph
        G = sample_graph(self.graphon_fn, self.n_nodes)
        # 2. Random initial temps in [0,1]
        init = np.random.rand(self.n_nodes).astype(float)
        # 3. Simulate
        traj = simulate_heat_diffusion(G, init, self.alpha, self.steps)  # shape (T+1, N)
        tensor = torch.from_numpy(traj).float()  # float32
        if self.transform:
            tensor = self.transform(tensor)
        return tensor
