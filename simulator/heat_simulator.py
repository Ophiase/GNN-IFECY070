# heat_simulator.py
from typing import Tuple
import networkx as nx
import numpy as np


def simulate_heat_diffusion(
    G: nx.Graph,
    initial_temps: np.ndarray,
    alpha: float,
    steps: int
) -> np.ndarray:
    """
    Simulate discrete-time heat diffusion on graph G.

    Temperature update:
        T_i(t+1) = T_i(t) + alpha * sum_{j in N(i)} [T_j(t) - T_i(t)]

    Args:
        G: undirected graph with n nodes
        initial_temps: shape (n,), initial node temperatures
        alpha: diffusion rate (0 < alpha <= 1/degree_max)
        steps: number of timesteps to simulate

    Returns:
        temps: array of shape (steps+1, n), temperatures at each timestep
    """
    n = G.number_of_nodes()
    temps = np.zeros((steps + 1, n), dtype=float)
    temps[0] = initial_temps.copy()

    neighbors = [list(G.adj[i]) for i in range(n)]

    for t in range(steps):
        T = temps[t]
        T_next = T.copy()
        for i in range(n):
            diff = sum(T[j] - T[i] for j in neighbors[i])
            T_next[i] += alpha * diff
        temps[t + 1] = T_next

    return temps
