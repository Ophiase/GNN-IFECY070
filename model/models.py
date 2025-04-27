# model/models.py
"""
Three GNN models for our phenomena datasets:
  - HeatGCN: node-level regression for heat diffusion
  - RWNode2Vec: node embeddings via random walks (Node2Vec)
  - PercGCN: node-level regression/classification for percolation component sizes
"""
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, Node2Vec


class HeatGCN(nn.Module):
    """
    GCN for predicting steady-state temperatures at each node.
    Input: x[i]=initial temperature
    Output: y_pred[i]=predicted final temperature
    """
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2"
        self.convs = nn.ModuleList()
        # input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        # hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        # output layer
        self.convs.append(GCNConv(hidden_channels, 1))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x.squeeze(-1)

class SkipGramRW(nn.Module):
    """
    Skip-Gram model for random-walk node embeddings.
    Learns two embedding tables (target & context) and uses
    noise-contrastive estimation (negative sampling).
    """
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int = 64,
    ):
        """
        Args:
            num_nodes: total number of nodes in the graph
            embedding_dim: dimensionality of the embeddings
        """
        super().__init__()
        self.target_emb = nn.Embedding(num_nodes, embedding_dim)
        self.context_emb = nn.Embedding(num_nodes, embedding_dim)

        # init
        nn.init.xavier_uniform_(self.target_emb.weight)
        nn.init.xavier_uniform_(self.context_emb.weight)

    def forward(
        self,
        centers: torch.LongTensor,       # shape (B,)
        contexts: torch.LongTensor,      # shape (B,)
        negatives: torch.LongTensor      # shape (B, K)
    ) -> torch.Tensor:
        """
        Compute the skip-gram loss for one batch.

        Returns the averaged NCE loss over the batch.
        """
        # embed
        v_c = self.target_emb(centers)                      # (B, D)
        v_o = self.context_emb(contexts)                    # (B, D)
        v_n = self.context_emb(negatives)                   # (B, K, D)

        # positive score (B,)
        score_pos = torch.sum(v_c * v_o, dim=1)
        log_pos = F.logsigmoid(score_pos)

        # negative scores (B, K)
        # for each negative sample, dot with center
        score_neg = torch.bmm(v_n, v_c.unsqueeze(2)).squeeze(2)  # (B, K)
        log_neg = F.logsigmoid(-score_neg).sum(1)                # sum over K

        # final NCE loss
        loss = - (log_pos + log_neg).mean()
        return loss

    def get_embeddings(self) -> torch.Tensor:
        """
        Return the learned target embeddings (num_nodes, embedding_dim).
        """
        return self.target_emb.weight.data

class RWNode2VecWrapper(nn.Module):
    """
    Wrapper around PyG's Node2Vec for learning node embeddings via random walks.
    """
    def __init__(
        self,
        edge_index,
        embedding_dim: int = 64,
        walk_length: int = 10,
        context_size: int = 5,
        walks_per_node: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
        sparse: bool = False,
        num_nodes: Optional[int] = None,
    ):
        super().__init__()
        self.model = Node2Vec(
            edge_index,
            embedding_dim,
            walk_length,
            context_size,
            walks_per_node=walks_per_node,
            p=p,
            q=q,
            num_negative_samples=num_negative_samples,
            sparse=sparse,
            num_nodes=num_nodes,
        )

    def forward(self, batch=None):
        return self.model(batch)

    def loss(self, pos_rw, neg_rw):
        return self.model.loss(pos_rw, neg_rw)

    def reset_parameters(self):
        self.model.reset_parameters()


class PercGCN(nn.Module):
    """
    GCN for predicting component size per node after bond percolation.
    Input: x[i]=1 (constant),
    Output: y_pred[i]=predicted component size (regression) or class.
    """
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2"
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, 1))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x.squeeze(-1)
