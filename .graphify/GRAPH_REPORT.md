# Graph Report - .  (2026-07-03)

## Corpus Check
- 45 files · ~1,99,250 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 67 nodes · 54 edges · 6 communities detected
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output
- Edge kinds: contains: 23 · method: 13 · rationale_for: 8 · calls: 6 · uses: 3 · inherits: 1


## Input Scope
- Requested: auto
- Resolved: all (source: default-auto)
- Included files: 45 · Candidates: recursive
- Excluded: 0 untracked · 0 ignored · 0 sensitive · 0 missing committed
- Warning: Git repository has no HEAD; falling back to all scope.
- Recommendation: Use --scope all or graphify.yaml inputs.corpus for a knowledge-base folder.
## God Nodes (most connected - your core abstractions)
1. `MultimodalDataset` - 7 edges
2. `EarlyFusionModel` - 5 edges
3. `UnimodalEncoder` - 5 edges
4. `LateFusionModel` - 5 edges
5. `CrossAttentionLayer` - 5 edges
6. `TextAnchoredCrossAttentionTransformer` - 5 edges
7. `Calculates research-standard evaluation metrics for CMU-MOSI:     - Mean Absolut` - 4 edges
8. `get_dataloaders()` - 3 edges
9. `calculate_metrics()` - 3 edges
10. `train_model()` - 3 edges

## Surprising Connections (you probably didn't know these)
- `Calculates research-standard evaluation metrics for CMU-MOSI:     - Mean Absolut` --uses--> `EarlyFusionModel`  [INFERRED]
  src/training/train.py → src/models/baseline_models.py
- `Calculates research-standard evaluation metrics for CMU-MOSI:     - Mean Absolut` --uses--> `LateFusionModel`  [INFERRED]
  src/training/train.py → src/models/baseline_models.py
- `Calculates research-standard evaluation metrics for CMU-MOSI:     - Mean Absolut` --uses--> `TextAnchoredCrossAttentionTransformer`  [INFERRED]
  src/training/train.py → src/models/transformer_fusion.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.25
Nodes (5): get_dataloaders(), MultimodalDataset, Loads the dataset from pickle file and returns PyTorch DataLoader objects      f, A standard PyTorch Dataset class for loading preprocessed, pre-aligned      mult, Dataset

### Community 1 - "Community 1"
Cohesion: 0.25
Nodes (4): LateFusionModel, Simple sequence encoder for a single modality., Late Fusion (Decision-Level Fusion) baseline model.     Processes each modality, UnimodalEncoder

### Community 2 - "Community 2"
Cohesion: 0.25
Nodes (4): CrossAttentionLayer, Our proposed state-of-the-art Text-Anchored Cross-Attention Transformer.     Use, A Cross-Attention layer where Modality A (e.g. Text) acts as the Query,     and, TextAnchoredCrossAttentionTransformer

### Community 3 - "Community 3"
Cohesion: 0.60
Nodes (4): calculate_metrics(), main(), Calculates research-standard evaluation metrics for CMU-MOSI:     - Mean Absolut, train_model()

### Community 4 - "Community 4"
Cohesion: 0.50
Nodes (2): EarlyFusionModel, Early Fusion (Feature-Level Fusion) baseline model.     Concatenates visual, aco

### Community 5 - "Community 5"
Cohesion: 1.00
Nodes (2): main(), run_experiment()

## Knowledge Gaps
- **7 isolated node(s):** `A standard PyTorch Dataset class for loading preprocessed, pre-aligned      mult`, `Loads the dataset from pickle file and returns PyTorch DataLoader objects      f`, `Early Fusion (Feature-Level Fusion) baseline model.     Concatenates visual, aco`, `Simple sequence encoder for a single modality.`, `Late Fusion (Decision-Level Fusion) baseline model.     Processes each modality` (+2 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 4`** (2 nodes): `EarlyFusionModel`, `Early Fusion (Feature-Level Fusion) baseline model.     Concatenates visual, aco`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 5`** (2 nodes): `main()`, `run_experiment()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Calculates research-standard evaluation metrics for CMU-MOSI:     - Mean Absolut` connect `Community 3` to `Community 4`, `Community 1`, `Community 2`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Why does `TextAnchoredCrossAttentionTransformer` connect `Community 2` to `Community 3`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Why does `LateFusionModel` connect `Community 1` to `Community 3`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **What connects `A standard PyTorch Dataset class for loading preprocessed, pre-aligned      mult`, `Loads the dataset from pickle file and returns PyTorch DataLoader objects      f`, `Early Fusion (Feature-Level Fusion) baseline model.     Concatenates visual, aco` to the rest of the system?**
  _7 weakly-connected nodes found - possible documentation gaps or missing edges._