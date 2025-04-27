# percolation_dataset.py
from typing import Optional
import torch
from torch.utils.data import Dataset
import networkx as nx
import numpy as np
from logic.graphon_generator import sample_graph, interp_graphon
from simulator.percolation_simulator import simulate_bond_percolation


class PercolationDataset(Dataset):
    """
    On-the-fly dataset of percolated graphs + component sizes.
    Returns a dict with 'graph' (adjacency matrix) and 'sizes' (component size vector).
    """

    def __init__(
        self,
        n_nodes: int,
        p: float,
        graphon_fn,
        resolution: int = 100,
        transform: Optional[callable] = None
    ):
        self.n_nodes = n_nodes
        self.p = p
        self.graphon_fn = graphon_fn
        self.resolution = resolution
        self.transform = transform

    def __len__(self) -> int:
        return 10_000

    def __getitem__(self, idx: int):
        G = sample_graph(self.graphon_fn, self.n_nodes)
        Gp = simulate_bond_percolation(G, self.p)
        # adjacency matrix
        A = nx.to_numpy_array(Gp, nodelist=range(self.n_nodes))
        # component sizes
        comps = nx.connected_components(Gp)
        sizes = np.zeros(self.n_nodes, dtype=int)
        for comp in comps:
            sz = len(comp)
            for node in comp:
                sizes[node] = sz
        sample = {
            'adj': torch.from_numpy(A).float(),
            'sizes': torch.from_numpy(sizes).long()
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
