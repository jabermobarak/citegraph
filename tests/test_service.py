import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from citegraph.api import create_app
from citegraph.service import Workspace


@pytest.fixture(scope="module")
def workspace():
    if not Path("data/graph.npz").exists():
        pytest.skip("Run python -m citegraph.cli prepare for real-data integration checks")
    return Workspace(Path("data"), Path("artifacts"))


@pytest.fixture(scope="module")
def client(workspace):
    with TestClient(create_app(Path("data"), Path("artifacts"))) as client:
        yield client


def test_live_predictions_match_committed_release(workspace):
    for name in ("gcn", "mlp"):
        with np.load(f"artifacts/{name}_predictions.npz", allow_pickle=False) as z:
            np.testing.assert_allclose(
                workspace.probabilities[name], z["probabilities"], rtol=1e-5, atol=1e-6
            )


def test_full_graph_api_and_neighborhood(client, workspace):
    overview = client.get("/api/overview").json()
    assert overview["nodes"] == 19717
    paper_id = overview["examples"][0]["id"]
    node = client.get(f"/api/node/{paper_id}").json()
    assert node["id"] == paper_id
    assert abs(sum(node["gcn"]["probabilities"]) - 1) < 1e-5
    graph = client.get(f"/api/neighborhood/{paper_id}").json()
    assert len(graph["nodes"]) <= 51
    actual = {
        tuple(sorted((str(workspace.graph.ids[a]), str(workspace.graph.ids[b]))))
        for a, b in workspace.graph.edges
    }
    assert all(tuple(sorted(pair)) in actual for pair in graph["edges"])
    assert client.get("/health").json()["status"] == "ready"
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_zero_edge_removal_reproduces_original(workspace):
    paper = workspace.overview()["examples"][0]["id"]
    result = workspace.stress(paper, 0, 2026)
    assert result["baseline"] == result["perturbed"]
    assert result["retained_undirected_edges"] == len(workspace.graph.edges)


def test_all_edges_removed_and_repeat_is_deterministic(workspace):
    paper = workspace.overview()["examples"][0]["id"]
    first = workspace.stress(paper, 1, 2026)
    second = workspace.stress(paper, 1, 2026)
    assert first["retained_undirected_edges"] == 0
    assert first["perturbed"] == second["perturbed"]


def test_invalid_inputs_and_metrics(client):
    assert client.get("/api/node/not-a-paper").status_code == 404
    assert client.post("/api/stress", json={"paper_id": "1", "drop_rate": 1.5}).status_code == 422
    assert client.post("/api/stress", json={"paper_id": "bad", "drop_rate": 0.5}).status_code == 422
    assert (
        client.post("/api/stress", json={"paper_id": "1", "drop_rate": 0.5, "unknown": 1}).status_code == 422
    )
    assert "citegraph_requests_total" in client.get("/metrics").text


def test_model_checksum_failure_precedes_deserialization(tmp_path, workspace):
    shutil.copytree("artifacts", tmp_path / "artifacts")
    path = tmp_path / "artifacts/gcn.safetensors"
    path.write_bytes(path.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="Model checksum"):
        Workspace(Path("data"), tmp_path / "artifacts")


def test_optional_authentication(monkeypatch, workspace):
    monkeypatch.setenv("CG_API_TOKEN", "local-test-token")
    with TestClient(create_app(Path("data"), Path("artifacts"))) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/overview").status_code == 401
        assert (
            client.get("/api/overview", headers={"Authorization": "Bearer local-test-token"}).status_code
            == 200
        )
