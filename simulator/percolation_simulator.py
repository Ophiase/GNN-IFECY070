# percolation_simulator.py
import networkx as nx
import numpy as np


def simulate_bond_percolation(
    G: nx.Graph,
    p: float
) -> nx.Graph:
    """
    Perform bond percolation: each edge is kept with probability p.

    Args:
        G: original graph
        p: retention probability [0,1]

    Returns:
        Gp: new graph with only the retained edges
    """
    Gp = nx.Graph()
    Gp.add_nodes_from(G.nodes())
    for u, v in G.edges():
        if np.random.rand() < p:
            Gp.add_edge(u, v)
    return Gp
