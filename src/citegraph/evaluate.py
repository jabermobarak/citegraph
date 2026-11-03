"""Independent temperature fitting, policy selection, and frozen final evaluation."""

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

from .data import SOURCE_SHA256, load
from .model import adjacency, fit

SEEDS = (11, 22, 33)
RELEASE_SEED = 11


def calibrate(logits: np.ndarray, labels: np.ndarray) -> float:
    def objective(log_temp):
        return log_loss(labels, softmax(logits / np.exp(log_temp), axis=1), labels=[0, 1, 2])

    result = minimize_scalar(objective, bounds=(-2.3, 2.3), method="bounded")
    return float(np.exp(result.x))


def choose_threshold(probabilities: np.ndarray, labels: np.ndarray, target: float = 0.9):
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    # Fixed grid, at least 30 policy cases; the target is a selection rule, not a guarantee.
    candidates = []
    for threshold in np.linspace(0, 0.99, 100):
        accepted = confidence >= threshold
        if accepted.sum() >= 30 and correct[accepted].mean() >= target:
            candidates.append(float(threshold))
    return min(candidates) if candidates else 1.01


def score(probabilities: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    accepted = confidence >= threshold
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        chosen = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if chosen.any():
            ece += chosen.mean() * abs((prediction[chosen] == y[chosen]).mean() - confidence[chosen].mean())
    return {
        "n": len(y),
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "nll": float(log_loss(y, probabilities, labels=[0, 1, 2])),
        "ece": float(ece),
        "coverage": float(accepted.mean()),
        "accepted_count": int(accepted.sum()),
        "accepted_accuracy": float((prediction[accepted] == y[accepted]).mean()) if accepted.any() else None,
        "accepted_errors": int((prediction[accepted] != y[accepted]).sum()),
        "confusion": confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist(),
    }


def train_release(data_dir: Path, output: Path, epochs: int = 200, seeds=SEEDS) -> dict:
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    graph = load(data_dir)
    x = torch.from_numpy(graph.x)
    a = adjacency(len(graph.ids), graph.edges)
    train, val = graph.splits["train"], graph.splits["validation"]
    temperature_idx, policy, test = (graph.splits[k] for k in ("temperature", "policy", "test"))
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "PubMed Diabetes citation network",
        "archive_sha256": SOURCE_SHA256,
        "nodes": len(graph.ids),
        "features": graph.x.shape[1],
        "undirected_edges": len(graph.edges),
        "split_counts": {k: len(v) for k, v in graph.splits.items()},
        "protocol": "Transductive: all node features and edges visible; disjoint labels for fitting/selection/calibration/policy/test",
        "seeds": list(seeds),
        "release_seed": RELEASE_SEED,
        "runs": [],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "torch_threads": 2,
        },
    }
    manifest = {
        "source_sha256": SOURCE_SHA256,
        "features": graph.x.shape[1],
        "hidden": 64,
        "release_seed": RELEASE_SEED,
        "models": {},
        "split_sha256": hashlib.sha256(b"".join(v.tobytes() for v in graph.splits.values())).hexdigest(),
    }
    for seed in seeds:
        for name, is_graph in (("gcn", True), ("mlp", False)):
            started = time.perf_counter()
            network, epoch, history = fit(
                x,
                a,
                torch.tensor(graph.y[train]),
                train,
                torch.tensor(graph.y[val]),
                val,
                seed,
                graph=is_graph,
                epochs=epochs,
            )
            with torch.inference_mode():
                logits = network(x, a).numpy()
            temperature = calibrate(logits[temperature_idx], graph.y[temperature_idx])
            probabilities = softmax(logits / temperature, axis=1)
            threshold = choose_threshold(probabilities[policy], graph.y[policy])
            row = {
                "seed": seed,
                "model": name,
                "best_epoch": epoch,
                "temperature": temperature,
                "review_threshold": threshold,
                "seconds": time.perf_counter() - started,
                "test": score(probabilities[test], graph.y[test], threshold),
                "uncalibrated_test": score(softmax(logits[test], axis=1), graph.y[test], threshold),
            }
            report["runs"].append(row)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "model": name,
                        "test_accuracy": row["test"]["accuracy"],
                        "epoch": epoch,
                        "seconds": row["seconds"],
                    }
                ),
                flush=True,
            )
            if seed == RELEASE_SEED:
                path = output / f"{name}.safetensors"
                save_file(network.state_dict(), str(path))
                manifest["models"][name] = {
                    "temperature": temperature,
                    "threshold": threshold,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                (output / f"{name}_training.json").write_text(json.dumps(history, indent=2))
                np.savez_compressed(
                    output / f"{name}_predictions.npz",
                    ids=graph.ids,
                    probabilities=probabilities.astype(np.float32),
                )
                if name == "gcn":
                    report["stress"] = {}
                    rng = np.random.default_rng(2026)
                    for drop in (0, 0.25, 0.5, 0.75, 1):
                        keep = rng.random(len(graph.edges)) >= drop
                        altered = adjacency(len(graph.ids), graph.edges[keep])
                        with torch.inference_mode():
                            stress_logits = network(x, altered).numpy()
                        report["stress"][str(drop)] = score(
                            softmax(stress_logits[test] / temperature, axis=1), graph.y[test], threshold
                        )
    counts = np.bincount(graph.y[train], minlength=3).astype(float)
    # The labeled training set is balanced; argmax deterministically resolves the majority tie to topic 0.
    majority = np.tile(counts / counts.sum(), (len(test), 1))
    report["majority"] = score(majority, graph.y[test], 0)
    report["summary"] = {}
    for name in ("gcn", "mlp"):
        rows = [r["test"] for r in report["runs"] if r["model"] == name]
        report["summary"][name] = {
            metric: {
                "mean": float(np.mean([r[metric] for r in rows])),
                "std": float(np.std([r[metric] for r in rows])),
            }
            for metric in ("accuracy", "macro_f1", "coverage", "ece")
        }
    # Paired bootstrap resamples nodes; graph dependence makes this descriptive only.
    gcn = np.load(output / "gcn_predictions.npz")["probabilities"][test].argmax(1)
    mlp = np.load(output / "mlp_predictions.npz")["probabilities"][test].argmax(1)
    correct_diff = (gcn == graph.y[test]).astype(float) - (mlp == graph.y[test]).astype(float)
    rng = np.random.default_rng(77)
    differences = [float(correct_diff[rng.integers(0, len(test), len(test))].mean()) for _ in range(500)]
    report["release_paired_accuracy_difference_ci95"] = np.quantile(differences, [0.025, 0.975]).tolist()
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest["report_sha256"] = hashlib.sha256((output / "report.json").read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [
        "# Measured results",
        "",
        "Real PubMed Diabetes citation data; custom split, not the standard Planetoid leaderboard.",
        "",
        "| Model | Mean accuracy | Std across seeds | Mean macro F1 |",
        "|---|---:|---:|---:|",
    ]
    for name, result in report["summary"].items():
        lines.append(
            f"| {name.upper()} | {result['accuracy']['mean']:.4f} | {result['accuracy']['std']:.4f} | {result['macro_f1']['mean']:.4f} |"
        )
    lines += [
        "",
        "3,000 test nodes; 60 training labels. Seeds 11, 22, 33. Release model is seed 11, selected in advance.",
        "All graph features and edges are visible during training; test labels are not used for learning or selection.",
        "Accuracy variation across initializations is not uncertainty across datasets. Nodes are correlated.",
        "Review thresholds target 90% accuracy on the separate policy set; test accepted accuracy is not guaranteed.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return report
