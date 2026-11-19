"""Spatial message-passing GNN followed by a temporal LSTM, using only PyTorch."""

import torch
from torch import nn
from torch.nn import functional as F
from graphs import DIRECTIONS


class SpatialCNN3D(nn.Module):
    """Encode field-local [channels, time, row, col] neighborhoods.

    The default (1, 5, 5) kernel follows the spatial encoder described in
    Miranda et al. (IGARSS 2024), cited by YieldSAT. Dense convolution and
    gathering at node coordinates is equivalent to applying this kernel to
    zero-padded patches, without materializing overlapping patches per node.
    Only existing nodes provide context; absent grid cells are zero padding.
    """

    def __init__(self, input_dim, output_dim=64, spatial_kernel=5, temporal_kernel=1,
                 max_grid_cells=65536):
        super().__init__()
        if min(input_dim, output_dim, max_grid_cells) < 1:
            raise ValueError("CNN dimensions and max_grid_cells must be positive")
        if any(k < 1 or k % 2 == 0 for k in (spatial_kernel, temporal_kernel)):
            raise ValueError("CNN kernel sizes must be positive odd integers")
        self.max_grid_cells = max_grid_cells
        self.conv = nn.Conv3d(input_dim, output_dim,
                              kernel_size=(temporal_kernel, spatial_kernel, spatial_kernel),
                              padding=(temporal_kernel // 2, spatial_kernel // 2, spatial_kernel // 2))
        # Normalize gathered nodes only: empty cells never enter these statistics.
        self.norm = nn.BatchNorm1d(output_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x, positions, graph_ptr, time_mask=None):
        if positions is None or positions.shape != (x.size(0), 2):
            raise ValueError("3D CNN requires positions [nodes, 2]")
        if positions.device != x.device or positions.dtype != torch.long:
            raise ValueError("3D CNN positions must be int64 on the input device")
        if graph_ptr is None or graph_ptr.ndim != 1 or graph_ptr.dtype != torch.long:
            raise ValueError("3D CNN requires an int64 graph_ptr delimiting field/year graphs")
        boundaries = graph_ptr.tolist()
        if (len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != x.size(0)
                or any(a >= b for a, b in zip(boundaries, boundaries[1:]))):
            raise ValueError("graph_ptr must partition all nodes into nonempty graphs")
        if time_mask is not None:
            if time_mask.shape != x.shape[:2] or time_mask.device != x.device:
                raise ValueError("3D CNN time_mask must have shape [nodes, time_steps] on the input device")
            # Empty observations cannot contribute their missing-value sentinel.
            x = x.masked_fill(~time_mask.bool().unsqueeze(-1), 0)
        encoded = []
        for start, end in zip(boundaries, boundaries[1:]):
            coordinates = positions[start:end]
            coordinates = coordinates - coordinates.amin(dim=0)
            height, width = (coordinates.amax(dim=0) + 1).tolist()
            if height * width > self.max_grid_cells:
                raise ValueError(
                    f"3D CNN field grid {height}x{width} exceeds cnn_max_grid_cells={self.max_grid_cells}; "
                    "check row/col grid coordinates or increase --cnn-max-grid-cells if memory permits")
            flat = coordinates[:, 0] * width + coordinates[:, 1]
            if flat.unique().numel() != end - start:
                raise ValueError("Duplicate pixel coordinates within a 3D CNN field/year graph")
            grid = x.new_zeros(height * width, x.size(1), x.size(2))
            grid = grid.index_copy(0, flat, x[start:end])
            grid = grid.permute(2, 1, 0).reshape(1, x.size(2), x.size(1), height, width)
            features = self.conv(grid)[0].flatten(start_dim=2)
            encoded.append(features.index_select(2, flat).permute(2, 1, 0))
        features = torch.cat(encoded, dim=0)
        values = features.reshape(-1, features.size(-1))
        if self.training and values.size(0) == 1:
            # A one-pixel, one-step graph cannot estimate batch variance.
            values = F.batch_norm(values, self.norm.running_mean, self.norm.running_var,
                                  self.norm.weight, self.norm.bias, training=False, eps=self.norm.eps)
        else:
            values = self.norm(values)
        return self.activation(values.reshape_as(features))


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
                 residual=False, aggregation="mean", temporal_pool="last", attention_mask=False,
                 cnn_hidden_dim=64, cnn_kernel_size=5, cnn_temporal_kernel=1, cnn_max_grid_cells=65536):
        super().__init__()
        if min(input_dim, hidden_dim, num_layers, gnn_hidden_dim, gnn_layers) < 1:
            raise ValueError("Model dimensions and layer counts must be positive")
        self.input_dim = input_dim
        if model_type not in ("gnn_lstm", "lstm", "3d_gnn_lstm", "3d_lstm"):
            raise ValueError("Unknown model_type")
        self.model_type = model_type
        has_gnn = model_type in ("gnn_lstm", "3d_gnn_lstm")
        if residual and not has_gnn:
            raise ValueError("Residual concatenation requires a GNN")
        self.residual = residual
        if aggregation not in ("mean", "directional") or temporal_pool not in ("last", "attention"):
            raise ValueError("Unknown aggregation or temporal_pool")
        if attention_mask and temporal_pool != "attention":
            raise ValueError("attention_mask requires attention pooling")
        if aggregation != "mean" and not has_gnn:
            raise ValueError("Directional aggregation requires a GNN")
        self.aggregation = aggregation
        self.temporal_pool = temporal_pool
        self.attention_mask = attention_mask
        self.cnn = (SpatialCNN3D(input_dim, cnn_hidden_dim, cnn_kernel_size,
                                cnn_temporal_kernel, cnn_max_grid_cells)
                    if model_type.startswith('3d_') else None)
        spatial_input_dim = cnn_hidden_dim if self.cnn is not None else input_dim
        layer_type = MessagePassingGNN if aggregation == "mean" else DirectionalGNN
        self.gnn = nn.ModuleList([
            layer_type(spatial_input_dim if i == 0 else gnn_hidden_dim, gnn_hidden_dim)
            for i in range(gnn_layers if has_gnn else 0)
        ])
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=(gnn_hidden_dim + (input_dim if residual else 0)) if self.gnn else spatial_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)
        # A shared scalar bias cancels in a temporal softmax, so omit it.
        self.attention = nn.Linear(hidden_dim, 1, bias=False) if temporal_pool == "attention" else None

    def forward(self, x, edge_index, positions=None, time_mask=None, return_attention=False, graph_ptr=None):
        if x.ndim != 3 or x.size(-1) != self.input_dim or min(x.shape[:2]) < 1:
            raise ValueError("x must have shape [nodes, time_steps, input_dim] with nonempty nodes/time")
        if edge_index.device != x.device:
            raise ValueError("x and edge_index must be on the same device")
        adjacency = mean_neighbor_adjacency(edge_index, x.size(0), x.dtype) if self.gnn else None
        if self.gnn and self.aggregation == "directional":
            adjacency = directional_adjacencies(adjacency, positions)
        original = x
        if self.cnn is not None:
            x = self.cnn(x, positions, graph_ptr, time_mask)
            if self.gnn and edge_index.numel():
                boundaries = graph_ptr.to(device=x.device)
                graph_ids = torch.repeat_interleave(torch.arange(len(boundaries) - 1, device=x.device),
                                                   boundaries[1:] - boundaries[:-1])
                if (graph_ids[edge_index[0]] != graph_ids[edge_index[1]]).any():
                    raise ValueError("GNN edges must not cross field/year graph boundaries")
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
