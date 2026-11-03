# Evaluation protocol

## Fixed graph and label splits

The PubMed Diabetes citation benchmark has 19,717 nodes and 500 precomputed TF-IDF word features. Each row is normalized by its feature sum. Raw source citations include reverse duplicates; conversion to unique, undirected, non-self connections yields 44,324 pairs. Symmetric arcs and self-loops are added for GCN normalization.

The split seed is 2026. Twenty nodes per class are selected for training. Remaining labels are split into stratified, disjoint groups:

| Split | Nodes | Permitted use |
|---|---:|---|
| Train | 60 | Supervised fitting |
| Validation | 500 | Checkpoint selection and early stopping |
| Temperature | 500 | Fit one temperature per model |
| Policy | 500 | Choose confidence review threshold |
| Final test | 3,000 | Frozen evaluation |
| Unlabeled remainder | 15,157 | Features/edges participate; labels unused |

This is transductive learning. Every node's fixed features and citation edges participate in graph propagation, including test nodes. Test labels do not enter the supervised loss, checkpoint selection, temperature fitting, or policy selection. Stratification uses labels to construct the benchmark partition. The source's TF-IDF representation was precomputed upstream, not refitted on this training split. This protocol must not be presented as unseen-paper or time-based generalization.

## Models

The GCN and MLP both have a 500→64→3 architecture with ReLU and 50% hidden dropout. Both use Adam with learning rate 0.02 and weight decay 0.0005, up to 200 epochs, early-stop patience 40, and minimum validation-loss improvement 0.00001. All settings were fixed before running the final comparison. The GCN adds normalized citation propagation at both layers; the MLP uses the same feature and label splits without propagation. Model initialization seeds are 11, 22, and 33. The release seed is 11, declared in code before evaluation, not chosen for its test score.

The training function accepts only training and validation label slices. A regression test mutates all other labels and checks that fitted weights and checkpoint selection remain identical. This guards the training contract; it does not prove that every possible experiment is free from leakage.

## Calibration and review

A positive scalar temperature minimizes NLL on the temperature split. Review policy selection searches confidence thresholds 0.00 to 0.99, accepting the lowest threshold that reaches at least 90% empirical accuracy on at least 30 policy nodes. If no threshold qualifies, the threshold is set above 1 and all nodes are reviewed. This is a heuristic operating-point rule, not a confidence bound or accuracy guarantee.

The release GCN threshold is 0.83. On final test nodes it accepts 1,101 of 3,000 nodes, with 91.8% accepted accuracy and 90 accepted errors. The MLP policy qualifies at no threshold in any of the three runs. Zero accepted errors at zero coverage is not successful automation. Human reviewer accuracy, corrections, and end-to-end benefit are not measured.

The report includes uncalibrated and calibrated NLL/ECE, accuracy, macro F1, confusion matrices, acceptance coverage, and accepted errors. Uncalibrated review statistics reuse calibrated-policy thresholds only as diagnostics; they are not separately selected operating points.

## Structural diagnostics

For release seed 11, random undirected citation pairs are removed at 0%, 25%, 50%, 75%, and 100%. Each graph is re-normalized, preserving self-loops, and the frozen GCN is evaluated with its original temperature and threshold. The weights are not refit. Masks are deterministic but not nested: each rate draws the next sample from a seeded RNG. Removing all links leaves self-connections, not a retrained MLP.

The interactive stress endpoint uses its explicit seed independently for each call; matching the rate alone does not guarantee matching the offline report's mask. The original graph display remains visible and is labeled as such. Removal is a diagnostic intervention on the model input, not a causal claim about scientific topics.

## Limits

One historical graph, one stratified label split, three initializations, no external dataset. Topic labels are coarse and may be imperfect. Connected nodes are not independent. The paired node-bootstrap interval is descriptive and can underestimate uncertainty due to graph dependence. Model seeds do not measure dataset-level uncertainty or statistical significance.

Words alone already predict many topics. Citations improve this particular comparison, but the app does not prove a general advantage for graph neural networks. The model does not ingest arbitrary text, discover new citations, estimate medical risk, or answer clinical questions. Titles and abstracts are absent from the provided benchmark.
