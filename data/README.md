# Data source

PubMed Diabetes citation network, distributed by the LINQS research group:

- Source archive: https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz
- Source description: https://linqs.org/datasets/
- SHA-256: `218e3c86146cf6d470b305d824f26c86845f0c858ad3fdd269f6cd2ce63e12bd`

The archive contains 19,717 paper IDs, three topic labels, 500 TF-IDF-weighted word features, and 44,338 directed citation records. The app canonicalizes reciprocal links into 44,324 unique undirected, non-self connections. The topic labels are Experimental Diabetes, Type 1 Diabetes, and Type 2 Diabetes. These describe scientific papers, not patients.

The loader reads only the named node and citation files. It does not fetch the full PubMed corpus, patient data, titles, or abstracts. Word features are upstream stemmed terms. The prepared cache is ignored by Git and must be created with `python -m citegraph.cli prepare`.

The source archive includes a descriptive README but no explicit dataset license. The raw archive and prepared feature/edge arrays are not redistributed here. Dataset use remains governed by the original source's terms; this project's MIT license covers original application code, not external data. Included fitted weights and prediction reports are generated experiment artifacts.

Dataset inspiration/reference: Namata, London, Getoor, and Huang, *Query-driven Active Surveying for Collective Classification*, 2012; see the original dataset collection for context. Graph-convolution reference: Kipf and Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR 2017.
