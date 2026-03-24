# Graph Connectivity Transformer (Adjacency-Matrix Prediction)

This project reproduces a focused, laptop-friendly subset of graph connectivity experiments using a 2-layer transformer over adjacency-matrix row tokens.

The task is matrix prediction:

- Input: self-loop-augmented adjacency matrix `A in {0,1}^{n x n}`.
- Target: connectivity matrix `R in {0,1}^{n x n}`, where `R[i,j]=1` iff `i` and `j` are in the same connected component.

All graphs are simple, undirected, and symmetric. Graph structure computations (connectivity, shortest paths, diameter) are done on adjacency matrices *without* self-loops.

## Implemented Experiments

1. **Baseline** (`ER(n=20, p=0.08)`, 2-layer transformer)
   - Train on ER
   - Evaluate on ER validation
   - OOD evaluate on `TwoChains` and `TwoCliques`
   - Run distance-conditioned capacity evaluation

2. **Capacity test**
   - Evaluate pairwise accuracy by shortest-path distance
   - Also compute cumulative accuracy over distances `<= d`
   - Infer maximum reliable path length (exact and cumulative) at threshold (default `0.99`)

3. **Restricted diameter**
   - Train on ER with filtering `diameter <= 9`
   - Re-evaluate ER and OOD suites
   - Compare OOD exact-match against baseline if baseline metrics are present

## Practical Notes

- This is a practical approximation (small-to-medium datasets) intended to run on CPU laptops.
- CUDA is supported automatically if available.
- Deterministic seeding is enabled for Python, NumPy, and PyTorch.

## Metrics

- **BCE loss**: entrywise `BCEWithLogitsLoss`.
- **Exact Match Accuracy**: a graph is correct only if all `n*n` entries match after thresholding logits at `0`.
- **Pairwise Accuracy**: entrywise accuracy over all matrix entries.
- **Distance-conditioned accuracy**:
  - Computed only on connected node pairs with finite shortest-path distance.
  - Disconnected pairs are reported separately.
  - Diagonal (`distance=0`) is excluded from path-length reliability analysis to avoid trivial inflation.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

From the `project/` directory:

### Baseline

```bash
python experiments/baseline.py
```

This script now builds a descriptive run folder based on the training configuration.

Outputs under `trainings/<run_id>/` (where `run_id` encodes the training config, e.g. `n20_d128_layers2_heads4_modeer_valer_ep20_seed42`):
- `best.pt`, `last.pt`
- `config.json`, `history.json`, `summary.json`
- `er_val_metrics.json`, `ood_metrics.json`, `distance_metrics.json`
- `epoch_*.pt` (per-epoch checkpoints)
- training and distance plots (the training-history plot includes a suptitle with model/dataset info)

If you want to retrain with different hyperparameters, change them in `experiments/baseline.py` and re-run; to force a fresh run delete the corresponding `trainings/<run_id>` folder.

### Capacity test

```bash
# Use a specific checkpoint (replace <run_id> with the one used for training)
python experiments/capacity_test.py --checkpoint trainings/<run_id>/best.pt
```

If you want it to train baseline automatically when missing:

```bash
python experiments/capacity_test.py --train_if_missing
```

By default, capacity-test writes its outputs under `runs/capacity_test/`. You can also use `--build_from_epochs` to build per-epoch distance metrics from `epoch_*.pt` saved in the training folder.

### Restricted diameter (`diameter <= 9`)

```bash
python experiments/restrict_diameter.py
```

Outputs under `runs/restrict_diameter/`, including baseline-vs-restricted OOD comparison plot if baseline exists.

## Expected Qualitative Behavior

- Unrestricted baseline should achieve strong in-distribution ER performance.
- OOD exact-match on `TwoChains` is typically much harder than ER (and often worse than `TwoCliques`).
- Restricted-diameter training (`<=9`) can improve OOD generalization.
- Distance-conditioned accuracy often degrades past path lengths near model capacity for `L=2` (around 9).
