import numpy as np
import pytest
import torch
from scipy.special import softmax

from citegraph.data import Graph
from citegraph.evaluate import calibrate, choose_threshold, score
from citegraph.model import adjacency, fit


def test_normalization_is_symmetric_with_isolated_node():
    edges = np.array([[0, 1], [1, 0], [0, 0]])
    a = adjacency(3, edges).to_dense().numpy()
    np.testing.assert_allclose(a, a.T)
    np.testing.assert_allclose(a, [[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 1]])


def test_invalid_edge_is_rejected():
    with pytest.raises(ValueError, match="Out-of-range"):
        adjacency(3, np.array([[0, 4]]))


def test_node_order_permutation_preserves_predictions():
    from citegraph.model import Network

    torch.manual_seed(7)
    model = Network(5, hidden=8).eval()
    x = torch.rand(4, 5)
    edges = np.array([[0, 1], [1, 2], [0, 3]])
    permutation = np.array([2, 0, 3, 1])
    inverse = np.argsort(permutation)
    with torch.no_grad():
        first = model(x, adjacency(4, edges))
        second = model(x[permutation], adjacency(4, inverse[edges]))
    torch.testing.assert_close(first[permutation], second)


def test_graph_split_overlap_is_rejected():
    graph = Graph(
        np.ones((3, 2)),
        np.arange(3),
        np.array([[0, 1]]),
        np.array(["1", "2", "3"]),
        ["a", "b"],
        {"train": np.array([0]), "test": np.array([0])},
    )
    with pytest.raises(ValueError, match="overlapping"):
        graph.validate()


def test_hidden_labels_cannot_change_training():
    torch.set_num_threads(2)
    rng = np.random.default_rng(2)
    x = torch.tensor(rng.normal(size=(30, 5)), dtype=torch.float32)
    edges = np.column_stack([np.arange(29), np.arange(1, 30)])
    a = adjacency(30, edges)
    y = np.arange(30) % 3
    train, val = np.arange(6), np.arange(6, 12)
    changed = y.copy()
    changed[12:] = (changed[12:] + 1) % 3
    first, epoch1, _ = fit(x, a, torch.tensor(y[train]), train, torch.tensor(y[val]), val, seed=5, epochs=12)
    second, epoch2, _ = fit(
        x, a, torch.tensor(changed[train]), train, torch.tensor(changed[val]), val, seed=5, epochs=12
    )
    assert epoch1 == epoch2
    for key in first.state_dict():
        torch.testing.assert_close(first.state_dict()[key], second.state_dict()[key], rtol=0, atol=0)


def test_temperature_calibration_reduces_nll():
    logits = np.array([[7, 0, 0], [0, 7, 0], [0, 0, 7], [7, 0, 0]], dtype=float)
    y = np.array([0, 1, 2, 1])
    temperature = calibrate(logits, y)
    assert (
        score(softmax(logits / temperature, axis=1), y, 0)["nll"]
        < score(softmax(logits, axis=1), y, 0)["nll"]
    )


def test_policy_abstains_when_no_threshold_meets_target():
    p = np.tile([0.8, 0.1, 0.1], (40, 1))
    assert choose_threshold(p, np.ones(40, dtype=int)) > 1
    result = score(p, np.ones(40, dtype=int), 1.01)
    assert result["coverage"] == 0
    assert result["accepted_accuracy"] is None


def test_review_policy_respects_minimum_sample_count():
    p = np.tile([0.6, 0.2, 0.2], (20, 1))
    assert choose_threshold(p, np.zeros(20, dtype=int)) > 1
