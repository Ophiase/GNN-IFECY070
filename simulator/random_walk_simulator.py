# random_walk_simulator.py
from typing import List
import networkx as nx
import numpy as np


def simulate_random_walk(
    G: nx.Graph,
    start_node: int,
    walk_length: int
) -> List[int]:
    """
    Perform a simple random walk of fixed length on G.

    Args:
        G: undirected graph with nodes labeled 0..n-1
        start_node: index to start
        walk_length: number of steps (edges); output list length = walk_length+1

    Returns:
        path: list of node indices visited
    """
    path = [start_node]
    current = start_node

    for _ in range(walk_length):
        neighbors = list(G.adj[current])
        if not neighbors:
            break  # dead end
        current = np.random.choice(neighbors)
        path.append(current)

    return path
