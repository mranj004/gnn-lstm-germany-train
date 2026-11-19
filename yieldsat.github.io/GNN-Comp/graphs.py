"""Grid graphs and disjoint graph batching for YieldSAT pixels."""

import numpy as np
import torch

# Relative source position minus destination position in image row/column axes.
DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def build_grid_graph(rows, cols, connectivity=8):
    """Connect existing neighboring pixels in ONE field/harvest-year image.

    Edges run in both directions. Missing pixels are not bridged. Self-loops
    are handled by a separate self transform, so isolated pixels remain valid nodes.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    coordinates = np.column_stack((rows, cols))
    if not np.isfinite(coordinates).all() or not np.equal(coordinates, np.floor(coordinates)).all():
        raise ValueError("row and col must contain finite integer grid coordinates")
    coordinates = coordinates.astype(np.int64)
    lookup = {tuple(point): i for i, point in enumerate(coordinates)}
    if len(lookup) != len(coordinates):
        raise ValueError("Duplicate pixel coordinates in a field/year; check graph grouping")
    offsets = [(1, 0), (0, 1)]
    if connectivity == 8:
        offsets += [(1, 1), (1, -1)]
    edges = []
    for i, (row, col) in enumerate(coordinates):
        for dr, dc in offsets:
            j = lookup.get((row + dr, col + dc))
            if j is not None:
                edges.extend(((i, j), (j, i)))
    return torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous()


def collate_graphs(graphs):
    """Concatenate whole graphs, offsetting edges without cross-graph edges."""
    if not graphs:
        raise ValueError("Cannot collate an empty batch")
    edges, indices, fields, years = [], [], [], []
    offset = 0
    graph_ptr = [0]
    for graph in graphs:
        edges.append(graph["edge_index"] + offset)
        offset += graph["sample"].size(0)
        graph_ptr.append(offset)
        indices.extend(graph["index"])
        fields.extend(graph["field_shared_name"])
        years.extend(graph["year"])
    return {
        "sample": torch.cat([g["sample"] for g in graphs]),
        "time_mask": torch.cat([g.get("time_mask", torch.ones(g["sample"].shape[:2], dtype=torch.bool)) for g in graphs]),
        "positions": torch.cat([g["positions"] for g in graphs]) if all("positions" in g for g in graphs) else None,
        "target": torch.cat([g["target"] for g in graphs]),
        "edge_index": torch.cat(edges, dim=1),
        "graph_ptr": torch.tensor(graph_ptr, dtype=torch.long),
        "index": indices, "field_shared_name": fields, "year": years,
    }
