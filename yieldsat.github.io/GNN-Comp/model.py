"""Spatial message-passing GNN followed by a temporal LSTM, using only PyTorch."""

import torch
from torch import nn
from graphs import DIRECTIONS


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


def directional_adjacencies(adjacency, positions):
    if positions is None or positions.shape != (adjacency.size(0), 2):
        raise ValueError("Directional aggregation requires positions [nodes, 2]")
    dst, src = adjacency.indices()
    offsets = positions[src] - positions[dst]
    if offsets.numel() and ((offsets.abs() > 1).any() or (offsets == 0).all(dim=1).any()):
        raise ValueError("Directional aggregation requires adjacent grid edges")
    matrices = []
    for direction in DIRECTIONS:
        selected = (offsets == offsets.new_tensor(direction)).all(dim=1)
        matrices.append(torch.sparse_coo_tensor(
            adjacency.indices()[:, selected], adjacency.values()[selected], adjacency.shape,
        ).coalesce())
    return matrices


class DirectionalGNN(nn.Module):
    """Direction-specific neighbor weights, divided by total distinct degree."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbor_linears = nn.ModuleList([nn.Linear(input_dim, output_dim, bias=False) for _ in DIRECTIONS])

    def forward(self, x, adjacencies):
        result = self.self_linear(x)
        for linear, adjacency in zip(self.neighbor_linears, adjacencies):
            neighbors = torch.sparse.mm(adjacency, x.flatten(start_dim=1)).reshape_as(x)
            result = result + linear(neighbors)
        return result


class GNN_LSTM_Model(nn.Module):
    """Predict one yield per pixel from [nodes, time_steps, input_dim].

    Batched graphs are a disjoint union: edge_index must not connect fields.
    The graph is static across the season; the LSTM models temporal changes.
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=1,
                 gnn_hidden_dim=32, gnn_layers=2, dropout=0.1, model_type="gnn_lstm",
                 residual=False, aggregation="mean", temporal_pool="last", attention_mask=False):
        super().__init__()
        if min(input_dim, hidden_dim, num_layers, gnn_hidden_dim, gnn_layers) < 1:
            raise ValueError("Model dimensions and layer counts must be positive")
        self.input_dim = input_dim
        if model_type not in ("gnn_lstm", "lstm"):
            raise ValueError("model_type must be gnn_lstm or lstm")
        self.model_type = model_type
        if residual and model_type == "lstm":
            raise ValueError("Residual concatenation requires a GNN")
        self.residual = residual
        if aggregation not in ("mean", "directional") or temporal_pool not in ("last", "attention"):
            raise ValueError("Unknown aggregation or temporal_pool")
        if attention_mask and temporal_pool != "attention":
            raise ValueError("attention_mask requires attention pooling")
        if aggregation != "mean" and model_type == "lstm":
            raise ValueError("Directional aggregation requires a GNN")
        self.aggregation = aggregation
        self.temporal_pool = temporal_pool
        self.attention_mask = attention_mask
        layer_type = MessagePassingGNN if aggregation == "mean" else DirectionalGNN
        self.gnn = nn.ModuleList([
            layer_type(input_dim if i == 0 else gnn_hidden_dim, gnn_hidden_dim)
            for i in range(gnn_layers if model_type == "gnn_lstm" else 0)
        ])
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=(gnn_hidden_dim + (input_dim if residual else 0)) if self.gnn else input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)
        # A shared scalar bias cancels in a temporal softmax, so omit it.
        self.attention = nn.Linear(hidden_dim, 1, bias=False) if temporal_pool == "attention" else None

    def forward(self, x, edge_index, positions=None, time_mask=None, return_attention=False):
        if x.ndim != 3 or x.size(-1) != self.input_dim or min(x.shape[:2]) < 1:
            raise ValueError("x must have shape [nodes, time_steps, input_dim] with nonempty nodes/time")
        if edge_index.device != x.device:
            raise ValueError("x and edge_index must be on the same device")
        adjacency = mean_neighbor_adjacency(edge_index, x.size(0), x.dtype) if self.gnn else None
        if self.gnn and self.aggregation == "directional":
            adjacency = directional_adjacencies(adjacency, positions)
        original = x
        for layer in self.gnn:
            x = self.dropout(torch.relu(layer(x, adjacency)))
        if self.residual:
            x = torch.cat([original, x], dim=-1)
        out, _ = self.lstm(x)
        weights = None
        context = out[:, -1, :]
        if self.attention is not None:
            scores = self.attention(out).squeeze(-1)
            if self.attention_mask:
                if time_mask is None or time_mask.shape != scores.shape:
                    raise ValueError("Masked attention requires time_mask [nodes, time_steps]")
                valid = time_mask.bool()
                scores = scores.masked_fill(~valid, -torch.inf)
                scores = torch.where(valid.any(dim=1, keepdim=True), scores, torch.zeros_like(scores))
                weights = torch.softmax(scores, dim=1) * valid
            else:
                weights = torch.softmax(scores, dim=1)
            context = (out * weights.unsqueeze(-1)).sum(dim=1)
        prediction = self.fc(context).squeeze(-1)
        return (prediction, weights) if return_attention else prediction
