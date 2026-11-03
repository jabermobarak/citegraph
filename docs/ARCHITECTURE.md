# Architecture

## Graph learning

Let X contain row-normalized word features and A the undirected adjacency matrix. Add self-loops, then compute S = D^(-1/2)(A + I)D^(-1/2). The implementation stores S as a coalesced sparse tensor.

The two-layer model computes:

`H = dropout(ReLU(S(XW1 + b1)))`

`logits = S(HW2 + b2)`

The feature-only MLP omits both S operations. This small implementation follows the graph-convolution idea in [Kipf and Welling, ICLR 2017](https://arxiv.org/abs/1609.02907); it is not an exact reproduction of their experimental protocol. In particular, this project uses its own split and checkpoint rules, and applies affine bias before propagation.

## Data and model integrity

Preparation requests one fixed public URL, checks the archive SHA-256, and reads the expected tar members in memory without extracting paths. It discards unused summary fields and normalizes node features. The local cache has a separate integrity manifest. Raw downloaded data is not committed.

Training saves safe tensor weights and data-only JSON/NPZ outputs. The release manifest binds each weight file and report to checksums, and binds the model to the source archive and split identities. Startup verifies those values before loading weights. It computes actual full-graph inference rather than serving committed probability arrays. Integration tests compare those live probabilities to the release outputs.

Checksums detect mismatches; they are not signatures or a trust boundary against a malicious person who controls both files and manifests.

## Service and UI

FastAPI loads the verified graph and both networks once during lifespan startup. Original predictions are cached in memory. Neighborhood inspection selects the root, up to 20 first-hop neighbors, then up to 30 second-hop nodes, using real source edges only. This display bound does not change the graph used for inference. The dataset's original IDs remain visible.

Stress requests validate a numeric ID, a finite drop rate in [0,1], and an integer seed. They rebuild graph normalization and run the frozen GCN under an inference lock. This serializes the heavier stress operation; it is not a high-throughput serving design. Request counters are process-local and reset on restart. Inputs are not persisted.

The frontend renders graph nodes directly on canvas, uses textContent for dataset fields, and has no external scripts, fonts, analytics, or API calls. Graph colors show predictions. Observed labels are displayed for training and test nodes to support inspection; labels from selection/calibration/policy/unlabeled partitions are hidden in the UI. The server does not expose a retraining endpoint.

The supported demo runs on localhost. The optional bearer token protects API data and metrics, while static UI, readiness, and API schema remain public. A shared deployment needs TLS, identity-based permissions, rate limits, monitoring, and compute quotas. Torch inference threads are set to two for a predictable small CPU workload.

## Deliberate scope

This is a real-data graph ML implementation and a local research tool. It does not claim distributed graph processing, a graph database, newly arriving-node inference, model explanations, clinical use, or production adoption. A practical next evaluation would use a temporal or inductive split and compare additional graph methods without changing the current held-out results.
