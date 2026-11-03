"""Frozen inference and bounded graph inspection, with verified artifacts."""

import hashlib
import json
import time
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.special import softmax

from .data import SOURCE_SHA256, TOPICS, load
from .model import Network, adjacency


class Workspace:
    def __init__(self, data_dir: Path, artifacts: Path):
        torch.set_num_threads(2)
        self.graph = load(data_dir)
        self.index = {paper: i for i, paper in enumerate(self.graph.ids)}
        self.artifacts = artifacts
        self.manifest = json.loads((artifacts / "manifest.json").read_text())
        manifest = self.manifest
        if manifest["source_sha256"] != SOURCE_SHA256:
            raise ValueError("Model and dataset source differ")
        split_hash = hashlib.sha256(b"".join(v.tobytes() for v in self.graph.splits.values())).hexdigest()
        if manifest["split_sha256"] != split_hash:
            raise ValueError("Model and dataset split differ")
        if hashlib.sha256((artifacts / "report.json").read_bytes()).hexdigest() != manifest["report_sha256"]:
            raise ValueError("Report checksum mismatch")
        self.report = json.loads((artifacts / "report.json").read_text())
        self.x = torch.from_numpy(self.graph.x)
        self.a = adjacency(len(self.graph.ids), self.graph.edges)
        self.models, self.probabilities = {}, {}
        for name, config in manifest["models"].items():
            path = artifacts / f"{name}.safetensors"
            if hashlib.sha256(path.read_bytes()).hexdigest() != config["sha256"]:
                raise ValueError("Model checksum mismatch")
            network = Network(manifest["features"], manifest["hidden"], graph=name == "gcn")
            network.load_state_dict(load_file(str(path)))
            network.eval()
            self.models[name] = network
            with torch.inference_mode():
                logits = network(self.x, self.a).numpy()
            self.probabilities[name] = softmax(logits / config["temperature"], axis=1)
        self.neighbors = [[] for _ in self.graph.ids]
        for left, right in self.graph.edges:
            self.neighbors[left].append(int(right))
            self.neighbors[right].append(int(left))
        self.partition = {}
        for name, indexes in self.graph.splits.items():
            self.partition.update({int(i): name for i in indexes})
        self.lock = Lock()

    def lookup(self, paper_id: str) -> int:
        if paper_id not in self.index:
            raise KeyError("Unknown paper ID")
        return self.index[paper_id]

    def prediction(self, index: int, name: str, probabilities: np.ndarray | None = None) -> dict:
        row = (self.probabilities[name] if probabilities is None else probabilities)[index]
        topic = int(row.argmax())
        confidence = float(row.max())
        return {
            "topic_id": topic,
            "topic": TOPICS[topic],
            "confidence": confidence,
            "probabilities": row.tolist(),
            "decision": "accept" if confidence >= self.manifest["models"][name]["threshold"] else "review",
        }

    def node(self, paper_id: str) -> dict:
        index = self.lookup(paper_id)
        partition = self.partition[index]
        known_label = TOPICS[self.graph.y[index]] if partition in {"train", "test"} else None
        words = np.argsort(-self.graph.x[index])[:10]
        return {
            "id": paper_id,
            "partition": partition,
            "degree": len(self.neighbors[index]),
            "observed_topic": known_label,
            "gcn": self.prediction(index, "gcn"),
            "mlp": self.prediction(index, "mlp"),
            "terms": [
                {"term": self.graph.words[i].removeprefix("w-"), "weight": float(self.graph.x[index, i])}
                for i in words
                if self.graph.x[index, i] > 0
            ],
        }

    def neighborhood(self, paper_id: str) -> dict:
        root = self.lookup(paper_id)
        first = sorted(self.neighbors[root], key=lambda i: (-len(self.neighbors[i]), i))[:20]
        second = []
        seen = {root, *first}
        for middle in first:
            for other in self.neighbors[middle]:
                if other not in seen:
                    second.append(other)
                    seen.add(other)
                if len(second) >= 30:
                    break
            if len(second) >= 30:
                break
        chosen = [root, *first, *second]
        chosen_set = set(chosen)
        edges = [
            [str(self.graph.ids[left]), str(self.graph.ids[right])]
            for left in chosen
            for right in self.neighbors[left]
            if right in chosen_set and left < right
        ]
        return {
            "root": paper_id,
            "nodes": [
                {
                    "id": str(self.graph.ids[i]),
                    "hop": 0 if i == root else (1 if i in first else 2),
                    "topic_id": self.prediction(i, "gcn")["topic_id"],
                    "degree": len(self.neighbors[i]),
                }
                for i in chosen
            ],
            "edges": edges,
            "truncated": len(self.neighbors[root]) > len(first),
            "max_nodes": 51,
        }

    def stress(self, paper_id: str, drop_rate: float, seed: int) -> dict:
        index = self.lookup(paper_id)
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        keep = rng.random(len(self.graph.edges)) >= drop_rate
        a = adjacency(len(self.graph.ids), self.graph.edges[keep])
        with self.lock, torch.inference_mode():
            logits = self.models["gcn"](self.x, a).numpy()
        probabilities = softmax(logits / self.manifest["models"]["gcn"]["temperature"], axis=1)
        return {
            "id": paper_id,
            "drop_rate": drop_rate,
            "seed": seed,
            "retained_undirected_edges": int(keep.sum()),
            "baseline": self.prediction(index, "gcn"),
            "perturbed": self.prediction(index, "gcn", probabilities),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "note": "Frozen weights and thresholds; a structural diagnostic, not a causal explanation.",
        }

    def overview(self) -> dict:
        test = self.graph.splits["test"]
        predictions = self.probabilities["gcn"].argmax(1)
        disagreement = test[predictions[test] != self.probabilities["mlp"][test].argmax(1)]
        errors = test[predictions[test] != self.graph.y[test]]
        examples = []
        for category, indexes in [
            ("Model disagreement", disagreement),
            ("GCN error", errors),
            ("Test examples", test),
        ]:
            for i in indexes[:4]:
                paper_id = str(self.graph.ids[i])
                if paper_id not in {e["id"] for e in examples}:
                    examples.append({"id": paper_id, "category": category})
        return {
            "nodes": len(self.graph.ids),
            "edges": len(self.graph.edges),
            "topics": TOPICS,
            "features": self.graph.x.shape[1],
            "splits": self.report["split_counts"],
            "summary": self.report["summary"],
            "runs": self.report["runs"],
            "majority": self.report["majority"],
            "stress": self.report["stress"],
            "release_seed": self.manifest["release_seed"],
            "examples": examples,
            "model_id": self.manifest["models"]["gcn"]["sha256"][:12],
            "thresholds": {name: row["threshold"] for name, row in self.manifest["models"].items()},
        }
