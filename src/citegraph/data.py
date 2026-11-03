"""Verified public-source loading. Archive members are read, never extracted."""

import hashlib
import json
import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
from sklearn.model_selection import train_test_split

SOURCE_URL = "https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz"
SOURCE_SHA256 = "218e3c86146cf6d470b305d824f26c86845f0c858ad3fdd269f6cd2ce63e12bd"
TOPICS = ["Experimental diabetes", "Type 1 diabetes", "Type 2 diabetes"]
SPLIT_SEED = 2026


@dataclass
class Graph:
    x: np.ndarray
    y: np.ndarray
    edges: np.ndarray
    ids: np.ndarray
    words: list[str]
    splits: dict[str, np.ndarray]

    def validate(self):
        n = len(self.ids)
        if self.x.shape != (n, len(self.words)) or self.y.shape != (n,):
            raise ValueError("Inconsistent node shapes")
        if len(set(self.ids.tolist())) != n or not np.isfinite(self.x).all() or (self.x < 0).any():
            raise ValueError("Invalid IDs or node features")
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError("Edges must have shape [E,2]")
        if self.edges.size and (self.edges.min() < 0 or self.edges.max() >= n):
            raise ValueError("Edge references a missing node")
        if not np.isin(self.y, [0, 1, 2]).all():
            raise ValueError("Unknown topic label")
        seen = set()
        for name, indexes in self.splits.items():
            current = set(indexes.tolist())
            if len(current) != len(indexes) or seen & current or any(i < 0 or i >= n for i in current):
                raise ValueError(f"Invalid or overlapping split: {name}")
            seen.update(current)


def split_nodes(y: np.ndarray) -> dict[str, np.ndarray]:
    # Twenty labels per topic, then disjoint stratified selection/calibration/policy/test sets.
    rng = np.random.default_rng(SPLIT_SEED)
    train = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:20] for c in range(3)])
    remaining = np.setdiff1d(np.arange(len(y)), train)
    splits = {"train": np.sort(train)}
    for name, count in [("validation", 500), ("temperature", 500), ("policy", 500), ("test", 3000)]:
        chosen, remaining = train_test_split(
            remaining, train_size=count, random_state=SPLIT_SEED, stratify=y[remaining]
        )
        splits[name] = np.sort(chosen)
    splits["unlabeled"] = np.sort(remaining)
    return splits


def prepare(directory: str | Path, archive: Path | None = None) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / "graph.npz"
    if cache.exists():
        load(directory)
        return cache
    if archive:
        raw = archive.read_bytes()
    else:
        with httpx.Client(follow_redirects=True, timeout=90) as client:
            response = client.get(SOURCE_URL)
            response.raise_for_status()
            raw = response.content
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Public archive checksum mismatch")
    with tarfile.open(fileobj=BytesIO(raw), mode="r:gz") as tar:
        lines = (
            tar.extractfile("Pubmed-Diabetes/data/Pubmed-Diabetes.NODE.paper.tab")
            .read()
            .decode()
            .splitlines()
        )
        header = lines[1].split("\t")
        words = [field.split(":")[1] for field in header if field.startswith("numeric:")]
        positions = {word: i for i, word in enumerate(words)}
        rows = [line.split("\t") for line in lines[2:] if line.strip()]
        ids = np.array([row[0] for row in rows])
        index = {paper: i for i, paper in enumerate(ids)}
        x = np.zeros((len(rows), len(words)), dtype=np.float32)
        y = np.empty(len(rows), dtype=np.int64)
        for i, row in enumerate(rows):
            y[i] = int(row[1].split("=")[1]) - 1
            for field in row[2:]:
                name, value = field.split("=", 1)
                if name in positions:
                    x[i, positions[name]] = float(value)
        edges = []
        for line in tar.extractfile("Pubmed-Diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab"):
            fields = line.decode().strip().split("\t")
            if len(fields) == 4:
                edges.append(
                    (index[fields[1].removeprefix("paper:")], index[fields[3].removeprefix("paper:")])
                )
    x /= np.maximum(x.sum(axis=1, keepdims=True), 1e-12)
    # Store each undirected, non-self pair once. Normalization later adds symmetric arcs and self-loops.
    edge_array = np.unique(np.sort(np.asarray(edges, dtype=np.int64), axis=1), axis=0)
    edge_array = edge_array[edge_array[:, 0] != edge_array[:, 1]]
    splits = split_nodes(y)
    graph = Graph(x, y, edge_array, ids, words, splits)
    graph.validate()
    if x.shape != (19717, 500):
        raise ValueError("Unexpected dataset dimensions")
    np.savez_compressed(cache, x=x, y=y, edges=edge_array, ids=ids, words=np.array(words), **splits)
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "archive_sha256": SOURCE_SHA256,
                "cache_sha256": digest,
                "nodes": len(ids),
                "edges": len(edge_array),
                "split_seed": SPLIT_SEED,
            },
            indent=2,
        )
    )
    return cache


def load(directory: str | Path) -> Graph:
    directory = Path(directory)
    path = directory / "graph.npz"
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest["archive_sha256"] != SOURCE_SHA256
        or manifest["cache_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError("Cached dataset integrity check failed")
    with np.load(path, allow_pickle=False) as z:
        splits = {
            name: z[name].copy()
            for name in ("train", "validation", "temperature", "policy", "test", "unlabeled")
        }
        graph = Graph(
            z["x"].copy(), z["y"].copy(), z["edges"].copy(), z["ids"].copy(), z["words"].tolist(), splits
        )
    graph.validate()
    return graph
