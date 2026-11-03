"""Small two-layer graph and feature-only networks implemented in plain PyTorch."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def adjacency(n: int, edges: np.ndarray) -> torch.Tensor:
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Invalid edges")
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("Out-of-range edge")
    pairs = np.unique(np.sort(edges, axis=1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    arcs = np.concatenate([pairs, pairs[:, ::-1], np.column_stack([np.arange(n), np.arange(n)])])
    degree = np.bincount(arcs[:, 0], minlength=n).astype(np.float32)
    values = 1 / np.sqrt(degree[arcs[:, 0]] * degree[arcs[:, 1]])
    return torch.sparse_coo_tensor(
        torch.from_numpy(arcs.T.copy()), torch.from_numpy(values), size=(n, n), check_invariants=True
    ).coalesce()


class Network(nn.Module):
    def __init__(self, features: int = 500, hidden: int = 64, graph: bool = True):
        super().__init__()
        self.first = nn.Linear(features, hidden)
        self.second = nn.Linear(hidden, 3)
        self.graph = graph

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        h = self.first(x)
        if self.graph:
            h = torch.sparse.mm(a, h)
        h = F.dropout(F.relu(h), p=0.5, training=self.training)
        result = self.second(h)
        if self.graph:
            result = torch.sparse.mm(a, result)
        return result


def fit(
    x: torch.Tensor,
    a: torch.Tensor,
    y_train: torch.Tensor,
    train: np.ndarray,
    y_validation: torch.Tensor,
    validation: np.ndarray,
    seed: int,
    graph: bool = True,
    epochs: int = 200,
    patience: int = 40,
):
    # Only the allowed train and validation label slices are accepted by this function.
    torch.manual_seed(seed)
    model = Network(x.shape[1], graph=graph)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)
    best_loss, best_state, best_epoch, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(x, a)
        loss = F.cross_entropy(output[train], y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(F.cross_entropy(model(x, a)[validation], y_validation))
        history.append({"epoch": epoch + 1, "train_loss": float(loss.detach()), "validation_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch = val_loss, epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch + 1 - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, history
