# 🌐 GNN-IFECY070
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Apache](https://img.shields.io/badge/License-Apache%202.0-D22128?style=for-the-badge&logo=apache&logoColor=white)  ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)  

This project is part of a course on graph neural networks. The main objectives are:  

- Model a physical phenomenon on realizations of a [Graphon](https://fr.wikipedia.org/wiki/Graphon).  
- Train a graph neural network to replicate this phenomenon.  
- Benchmark the results.  
- Document findings in a report.  

### TODO  

- $\checkmark$ Graphon generators (image generators)  
- $\checkmark$ Graphon samplers  
    - $\mathbb{R}^2$ coordinates?  
    - Discretization?  
- $\checkmark$ Visualize sampled graphons  
- Sampling benchmark:  
    - Density, clustering, etc.  
- Phenomena to model:  
    - **Valid options:**  
        - Heat (or Diffusion) Dynamics  
        - Random Walks & Hitting Times  
        - Epidemic Spread (SI/SIR/SIS Models)  
        - Synchronization of Oscillators (Kuramoto Model)  
        - Electrical Resistor Network  
        - Percolation & Giant-Component Formation  
        - Reaction-Diffusion on Networks  
        - Predicting phenomena  
        - Navier-Stokes Flow (pressure-based)  
        - Electrostatic Potentials (Coulomb’s Law)  
        - Fractional Laplacian Diffusion  
- Implement Graph Neural Network  
- Benchmark and analyze results  
- Write the final report  

## Other

Datasets
- Heat Diffusion: 
    - Graph $\to$ Node
        - Possible architectures: GCN, GraphSAGE, GAT
    - Predict each node’s steady‐state temperature (continuous).
- Random-Walk Visitation: 
    - Graph $\to$ Sequence
        - Possible architectures: Sequence‐GNN (GGNN), PointerNet
    - Generate length-L node visitation sequences.
- Bond Percolation: Graph $\to$ Node
    - Grah $\to$ Node
        - Possible architectures: GCN + node‐regressor MLP
    - Assign each node its connected-component size (integer).