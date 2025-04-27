# random_walk_dataset.py
from typing import Optional
import torch
from torch.utils.data import Dataset
import networkx as nx
import numpy as np
from logic.graphon_generator import sample_graph, interp_graphon
from simulator.random_walk_simulator import simulate_random_walk


class RandomWalkDataset(Dataset):
    """
    On-the-fly dataset of random-walk paths on graphon graphs.
    Each sample is a tensor of shape (L+1,) of node indices.
    """

    def __init__(
        self,
        n_nodes: int,
        walk_length: int,
        graphon_fn,
        resolution: int = 100,
        transform: Optional[callable] = None
    ):
        self.n_nodes = n_nodes
        self.walk_length = walk_length
        self.graphon_fn = graphon_fn
        self.resolution = resolution
        self.transform = transform

    def __len__(self) -> int:
        return 10_000

    def __getitem__(self, idx: int) -> torch.Tensor:
        G = sample_graph(self.graphon_fn, self.n_nodes)
        start = np.random.randint(self.n_nodes)
        path = simulate_random_walk(G, start, self.walk_length)
        tensor = torch.tensor(path, dtype=torch.long)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor
