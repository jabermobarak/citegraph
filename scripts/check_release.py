"""Lightweight artifact and measured-result gates, with no retraining."""

import json
from pathlib import Path

from citegraph.service import Workspace

workspace = Workspace(Path("data"), Path("artifacts"))
report = workspace.report
assert report["nodes"] == 19717
assert report["split_counts"]["train"] == 60
assert report["split_counts"]["test"] == 3000
assert report["summary"]["gcn"]["accuracy"]["mean"] > 0.65
assert len(report["runs"]) == 6
print(json.dumps({"verified": True, "models": list(workspace.models), "nodes": report["nodes"]}))
