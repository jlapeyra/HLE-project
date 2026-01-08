from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv
from torch_geometric.utils import negative_sampling


# =========================
# 1) NetworkX -> PyG helpers
# =========================

def _as_float_tensor(x: Any, *, device: torch.device) -> torch.Tensor:
    """Convert scalar/list/np.ndarray/torch.Tensor to float32 tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    arr = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)

def _get_node_feat(G: nx.Graph, n: Any, key: str, feat_dim: int) -> np.ndarray:
    v = G.nodes[n].get(key, None)
    if v is None:
        return np.zeros((feat_dim,), dtype=np.float32)
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    if a.size != feat_dim:
        raise ValueError(f"Node {n} has {key} dim {a.size}, expected {feat_dim}")
    return a

def _get_edge_feat(G: nx.Graph, u: Any, v: Any, key: str, feat_dim: int) -> np.ndarray:
    data = G.get_edge_data(u, v, default=None)
    if data is None:
        return np.zeros((feat_dim,), dtype=np.float32)

    # For MultiGraph, edge_data is a dict-of-dicts; for Graph it's a dict.
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        # Take the first parallel edge's attributes by default
        first_k = next(iter(data.keys()))
        attrs = data[first_k]
    else:
        attrs = data

    val = attrs.get(key, None)
    if val is None:
        return np.zeros((feat_dim,), dtype=np.float32)

    a = np.asarray(val, dtype=np.float32).reshape(-1)
    if a.size != feat_dim:
        raise ValueError(f"Edge ({u},{v}) has {key} dim {a.size}, expected {feat_dim}")
    return a

def infer_feature_dims(
    pairs: List[Tuple[nx.Graph, nx.Graph]],
    node_key: str = "x",
    edge_key: str = "edge_attr",
) -> Tuple[int, int]:
    """Infer node/edge feature dimensions from the first occurrence in data."""
    node_dim = None
    edge_dim = None

    for Gin, Gout in pairs:
        for G in (Gin, Gout):
            # node dim
            for n, attrs in G.nodes(data=True):
                if node_key in attrs:
                    node_dim = int(np.asarray(attrs[node_key]).reshape(-1).shape[0])
                    break
            # edge dim
            if edge_dim is None:
                for u, v, attrs in G.edges(data=True):
                    if edge_key in attrs:
                        edge_dim = int(np.asarray(attrs[edge_key]).reshape(-1).shape[0])
                        break
            if node_dim is not None and edge_dim is not None:
                return node_dim, edge_dim

    if node_dim is None:
        raise ValueError(f"Could not infer node feature dim: no node attribute '{node_key}' found.")
    if edge_dim is None:
        raise ValueError(f"Could not infer edge feature dim: no edge attribute '{edge_key}' found.")
    return node_dim, edge_dim

def nx_to_pyg_data(
    G: nx.Graph,
    node_order: List[Any],
    node_dim: int,
    edge_dim: int,
    node_key: str = "x",
    edge_key: str = "edge_attr",
    directed: bool = False,
) -> Data:
    """
    Convert a NetworkX graph to a PyG Data object.
    Assumes node features in G.nodes[n][node_key] and edge features in G[u][v][edge_key].
    Missing features -> zeros.
    """
    idx: Dict[Any, int] = {n: i for i, n in enumerate(node_order)}

    # Node features
    X = np.stack([_get_node_feat(G, n, node_key, node_dim) for n in node_order], axis=0)

    # Edges
    edges_u: List[int] = []
    edges_v: List[int] = []
    E: List[np.ndarray] = []

    for u, v in G.edges():
        if u not in idx or v not in idx:
            raise ValueError("Graph contains nodes not in node_order (alignment required).")
        iu, iv = idx[u], idx[v]
        edges_u.append(iu); edges_v.append(iv)
        E.append(_get_edge_feat(G, u, v, edge_key, edge_dim))
        # For undirected graphs in PyG, you usually add both directions:
        if not directed:
            edges_u.append(iv); edges_v.append(iu)
            E.append(_get_edge_feat(G, u, v, edge_key, edge_dim))

    if len(E) == 0:
        # Allow edgeless graphs
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_dim), dtype=torch.float32)
    else:
        edge_index = torch.tensor([edges_u, edges_v], dtype=torch.long)
        edge_attr = torch.from_numpy(np.stack(E, axis=0)).to(dtype=torch.float32)

    return Data(
        x=torch.from_numpy(X).to(dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(node_order),
    )


# =========================
# 2) Dataset wrapper
# =========================

class NxPairDataset(torch.utils.data.Dataset):
    """
    Wraps: data: list[(Gin: nx.Graph, Gout: nx.Graph)]
    Assumes aligned nodes: same node IDs in Gin and Gout (or at least in a shared union),
    and you want fixed node identities across input/output.
    """
    def __init__(
        self,
        pairs: List[Tuple[nx.Graph, nx.Graph]],
        node_key: str = "x",
        edge_key: str = "edge_attr",
        directed: bool = False,
    ):
        self.pairs = pairs
        self.node_key = node_key
        self.edge_key = edge_key
        self.directed = directed
        self.node_dim, self.edge_dim = infer_feature_dims(pairs, node_key=node_key, edge_key=edge_key)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> Tuple[Data, Data]:
        Gin, Gout = self.pairs[i]

        # Node alignment: choose a deterministic order over the union (or intersection).
        # Union is safest if nodes may appear isolated in either graph.
        node_order = sorted(set(Gin.nodes()).union(set(Gout.nodes())), key=str)

        din = nx_to_pyg_data(
            Gin, node_order=node_order,
            node_dim=self.node_dim, edge_dim=self.edge_dim,
            node_key=self.node_key, edge_key=self.edge_key,
            directed=self.directed,
        )
        dout = nx_to_pyg_data(
            Gout, node_order=node_order,
            node_dim=self.node_dim, edge_dim=self.edge_dim,
            node_key=self.node_key, edge_key=self.edge_key,
            directed=self.directed,
        )
        return din, dout


# =========================
# 3) Graph->Graph model (aligned nodes, changing edges)
# =========================

class Graph2Graph(nn.Module):
    def __init__(self, x_dim: int, e_dim: int, hidden: int = 128, layers: int = 3):
        super().__init__()

        self.x_proj = nn.Linear(x_dim, hidden)
        self.e_proj = nn.Linear(e_dim, hidden)

        def mlp():
            return nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )

        self.convs = nn.ModuleList([GINEConv(nn=mlp(), edge_dim=hidden) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

        # Node feature head
        self.node_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, x_dim),
        )

        # Edge existence + edge feature heads (queried edges)
        self.edge_exist_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.edge_feat_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, e_dim),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        h = self.x_proj(x)
        e = self.e_proj(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index, e)
            h = norm(h)
            h = F.relu(h)
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index_query: torch.Tensor,
        edge_attr_query: torch.Tensor,
    ):
        h = self.encode(x, edge_index, edge_attr)
        node_out = self.node_head(h)

        src, dst = edge_index_query
        pair = torch.cat(
            [h[src], h[dst], self.e_proj(edge_attr_query)],
            dim=-1,
        )

        edge_logits = self.edge_exist_head(pair).squeeze(-1)
        edge_out = self.edge_feat_head(pair)

        return node_out, edge_out, edge_logits


# =========================
# 4) Training (structure + node feats + edge feats)
# =========================

def build_edge_queries(
    data_in: Data,
    data_out: Data,
    num_neg: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Query edges = all positive edges from target output graph + negative samples.
    Returns (edge_index_q, edge_attr_q, edge_label, num_pos)
    """
    device = data_in.x.device
    pos_edge_index = data_out.edge_index.to(device)
    num_pos = pos_edge_index.size(1)

    if num_neg is None:
        num_neg = num_pos

    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=int(data_in.num_nodes),
        num_neg_samples=num_neg,
        method="sparse",
    ).to(device)

    edge_index_q = torch.cat([pos_edge_index, neg_edge_index], dim=1)

    # edge_attr for queried edges:
    # - positives use target attributes (for edge feature regression)
    # - negatives are zeros
    e_dim = int(data_out.edge_attr.size(-1))
    pos_edge_attr = data_out.edge_attr.to(device)
    neg_edge_attr = torch.zeros((neg_edge_index.size(1), e_dim), device=device)
    edge_attr_q = torch.cat([pos_edge_attr, neg_edge_attr], dim=0)

    edge_label = torch.cat(
        [torch.ones(num_pos, device=device), torch.zeros(neg_edge_index.size(1), device=device)],
        dim=0,
    )
    return edge_index_q, edge_attr_q, edge_label, num_pos


def train(
    pairs: List[Tuple[nx.Graph, nx.Graph]],
    *,
    node_key: str = "x",
    edge_key: str = "edge_attr",
    directed: bool = False,
    batch_size: int = 8,
    epochs: int = 50,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    hidden: int = 128,
    layers: int = 3,
    w_node: float = 1.0,
    w_struct: float = 1.0,
    w_edge: float = 1.0,
    device: Optional[str] = None,
) -> Graph2Graph:
    """
    Full training loop.
    - aligned nodes
    - predicts node features, edge existence, and edge features (for edges that exist in target)
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = NxPairDataset(pairs, node_key=node_key, edge_key=edge_key, directed=directed)

    # IMPORTANT: loader must not try to "merge" (data_in, data_out) tuples into a single Data object
    # PyG's DataLoader can collate tuples; it will return (Batch, Batch) for each step.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Graph2Graph(x_dim=dataset.node_dim, e_dim=dataset.edge_dim, hidden=hidden, layers=layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for data_in, data_out in loader:
            data_in = data_in.to(dev)
            data_out = data_out.to(dev)

            edge_index_q, edge_attr_q, edge_label, num_pos = build_edge_queries(data_in, data_out)

            pred_x, pred_e, edge_logits = model(
                x=data_in.x,
                edge_index=data_in.edge_index,
                edge_attr=data_in.edge_attr,
                edge_index_query=edge_index_q,
                edge_attr_query=edge_attr_q,
            )

            # Node feature regression (all nodes)
            loss_node = F.mse_loss(pred_x, data_out.x)

            # Structure prediction (edge existence)
            loss_struct = F.binary_cross_entropy_with_logits(edge_logits, edge_label)

            # Edge feature regression on positives only
            loss_edge = F.mse_loss(pred_e[:num_pos], data_out.edge_attr)

            loss = w_node * loss_node + w_struct * loss_struct + w_edge * loss_edge

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            running += float(loss.item())

        avg = running / max(1, len(loader))
        print(f"Epoch {epoch:03d} | loss {avg:.4f}")

    return model


# =========================
# Example call
# =========================
if __name__ == "__main__":
    # Your data:
    # data: list[tuple[nx.Graph, nx.Graph]] = ...

    model = train(data, epochs=100, batch_size=4)
    print("Ready: call train(data) with your list[(Gin,Gout)].")
