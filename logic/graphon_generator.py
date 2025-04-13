from typing import Callable
from noise import pnoise2
import numpy as np
import networkx as nx

GraphonFunction = Callable[[float, float], float]

###################################################################################


def generate_constant_graphon(p: float, resolution: int = 100) -> np.ndarray:
    return np.full((resolution, resolution), p, dtype=float)


def generate_piecewise_graphon(k: int, resolution: int = 100) -> np.ndarray:
    grid = np.zeros((resolution, resolution), dtype=float)
    step = resolution // k
    for i in range(k):
        for j in range(k):
            value = np.random.uniform(0, 1)
            grid[i*step:(i+1)*step, j*step:(j+1)*step] = value
    return grid


def generate_min_graphon(resolution: int = 100) -> np.ndarray:
    x = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, x)
    return np.minimum(X, Y)


def generate_perlin_graphon(scale: float = 10.0, resolution: int = 100) -> np.ndarray:
    grid = np.zeros((resolution, resolution), dtype=float)
    for i in range(resolution):
        for j in range(resolution):
            grid[i, j] = pnoise2(i/scale, j/scale, octaves=6, persistence=0.5,
                                 lacunarity=2.0, repeatx=1024, repeaty=1024, base=0)
    grid = (grid - grid.min()) / (grid.max() - grid.min())
    return grid

###################################################################################


def interp_graphon(W: np.ndarray) -> GraphonFunction:
    resolution = W.shape[0]

    def fn(x: float, y: float) -> float:
        i = min(int(x * (resolution - 1)), resolution - 2)
        j = min(int(y * (resolution - 1)), resolution - 2)
        dx = x * (resolution - 1) - i
        dy = y * (resolution - 1) - j
        a = W[i, j]
        b = W[i+1, j]
        c = W[i, j+1]
        d = W[i+1, j+1]
        return (a * (1-dx) * (1-dy) + b * dx * (1-dy) + c * (1-dx) * dy + d * dx * dy)
    return fn


def sample_graph(W_fn: GraphonFunction, n: int) -> nx.Graph:
    xs = np.random.rand(n)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if np.random.rand() < W_fn(xs[i], xs[j]):
                G.add_edge(i, j)
    return G
