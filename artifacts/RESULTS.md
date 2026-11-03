# Measured results

Real PubMed Diabetes citation data; custom split, not the standard Planetoid leaderboard.

| Model | Mean accuracy | Std across seeds | Mean macro F1 |
|---|---:|---:|---:|
| GCN | 0.7480 | 0.0018 | 0.7513 |
| MLP | 0.6849 | 0.0021 | 0.6908 |

3,000 test nodes; 60 training labels. Seeds 11, 22, 33. Release model is seed 11, selected in advance.
All graph features and edges are visible during training; test labels are not used for learning or selection.
Accuracy variation across initializations is not uncertainty across datasets. Nodes are correlated.
Review thresholds target 90% accuracy on the separate policy set; test accepted accuracy is not guaranteed.
