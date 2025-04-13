
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from logic.graphon_generator import GraphonFunction, sample_graph


def display_graphon(W: np.ndarray, title: str) -> None:
    plt.figure(figsize=(4, 4))
    plt.imshow(W, cmap='viridis', origin='lower')
    plt.title(title)
    plt.colorbar()
    plt.show()


def display_sampled_graphs(W_fn: GraphonFunction, title: str, n: int = 20, samples: int = 3) -> None:
    fig, axes = plt.subplots(1, samples, figsize=(4 * samples, 4))
    for ax in axes:
        G = sample_graph(W_fn, n)
        pos = nx.spring_layout(G)
        nx.draw(G, pos=pos, node_size=20, ax=ax,
                node_color='skyblue', edge_color='gray')
        # nx.draw(G, ax=ax, node_size=20, with_labels=False, node_color='skyblue', edge_color='gray')
        ax.set_title(title)
    plt.show()
