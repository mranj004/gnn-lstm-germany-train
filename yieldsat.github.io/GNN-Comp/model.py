"""Spatial message-passing GNN followed by a temporal LSTM, using only PyTorch."""

import torch
from torch import nn


def mean_neighbor_adjacency(edge_index, num_nodes, dtype):
    """Average distinct incoming neighbors, excluding self; edges are [src, dst]."""
    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must be a long tensor of shape [2, E]")
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
        raise ValueError("edge_index contains a node outside the input graph")
    edges = edge_index[:, edge_index[0] != edge_index[1]]
    adjacency = torch.sparse_coo_tensor(
        edges.flip(0), torch.ones(edges.size(1), device=edges.device, dtype=dtype),
        (num_nodes, num_nodes),
    ).coalesce()
    row = adjacency.indices()[0]
    # Count each neighbor once; a node's own features have a separate update.
    values = torch.ones_like(adjacency.values())
    degree = torch.zeros(num_nodes, device=edges.device, dtype=dtype)
    degree.index_add_(0, row, values)
    return torch.sparse_coo_tensor(
        adjacency.indices(), values / degree.clamp_min(1)[row], adjacency.shape,
    ).coalesce()


class MessagePassingGNN(nn.Module):
    """Combine own features and mean neighbor messages at every time step."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbor_linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x, adjacency):
        neighbors = torch.sparse.mm(adjacency, x.flatten(start_dim=1)).reshape_as(x)
        return self.self_linear(x) + self.neighbor_linear(neighbors)


class GNN_LSTM_Model(nn.Module):
    """Predict one yield per pixel from [nodes, time_steps, input_dim].

    Batched graphs are a disjoint union: edge_index must not connect fields.
    The graph is static across the season; the LSTM models temporal changes.
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=1,
                 gnn_hidden_dim=32, gnn_layers=2, dropout=0.1):
        super().__init__()
        if min(input_dim, hidden_dim, num_layers, gnn_hidden_dim, gnn_layers) < 1:
            raise ValueError("Model dimensions and layer counts must be positive")
        self.input_dim = input_dim
        self.gnn = nn.ModuleList([
            MessagePassingGNN(input_dim if i == 0 else gnn_hidden_dim, gnn_hidden_dim)
            for i in range(gnn_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=gnn_hidden_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        if x.ndim != 3 or x.size(-1) != self.input_dim or min(x.shape[:2]) < 1:
            raise ValueError("x must have shape [nodes, time_steps, input_dim] with nonempty nodes/time")
        if edge_index.device != x.device:
            raise ValueError("x and edge_index must be on the same device")
        adjacency = mean_neighbor_adjacency(edge_index, x.size(0), x.dtype)
        for layer in self.gnn:
            x = self.dropout(torch.relu(layer(x, adjacency)))
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)
