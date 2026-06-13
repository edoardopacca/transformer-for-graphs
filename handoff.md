# Handoff — Transformer for Graphs (n=40 BIG runs)

## Goal we're working toward
The paper "Transformers Provably Learn Algorithmic Solutions for Graph Connectivity, But Only with the Right Data" suggests that for disentangled transformers, the training works much bettere if in the training set there are only graphs with D=diameter <= 3^L where L is the number of layers. My professor asked me to try different training with different graphs (ER unfiltered, 2chains,2cliques,ER(d<=7,<=9,<=11), curriculum (in a set <=11)) and test them in distribution to see how the training is. I'm also testing them OOD to test the models. Read the main.tex or main.pdf to see all the report.
What we recently did was train a transformer on graph connectivity (ER, n=40, p=0.05) across four diameter filters (unfiltered, D≤11, D≤9, D≤7) plus a curriculum variant, using the **BIG** setup (`d_model=512`, normalized-ReLU attention, online data generation, bf16, batch=1000, 1M steps). Then run an **OOD evaluation** of every trained model against 2chains, 2cliques, and unfiltered-ER test sets (with per-diameter-bucket breakdown). Goal: close the project, write up the report, send to the prof this weekend, meet early next week.

## Current state of the code

- `model.py`: `ModelConfig.attn_kind` switches between `"softmax"` and `"normalized_relu"`. BIG runs use `normalized_relu`.
- `data.py`: APSP uses `scipy.sparse.csgraph.shortest_path` (the Python BFS version was 6–10× slower; this is the patch that made D≤7 tractable).
- `experiments2/retrain_and_test_er_n40_big.py`: BIG training script. Online IterableDataset, cosine LR with warmup, bf16 autocast, eval every 5000 steps on a pre-generated 10k test set.
- `experiments2/curriculum_er_n40_big.py`: 4-phase curriculum (D≤7 → D≤9 → D≤10 → D≤11) with loss-threshold phase transitions.
- `experiments2/ood_evaluation.py`: reads `attn_kind` from checkpoint config (fix from earlier).
- All sbatch scripts under `scripts/`; the 4 BIG retraining sbatches and the OOD sbatch auto-detect checkpoints from `runs/report2/retrain_er_n40_big_<JOBID>/...`.

## Files actively edited this session

- `scripts/retrain_er_n40_big_diam9.sbatch` — moved partition `long_gpuh200` → `gpuh200` (kept 24 CPU).
- `scripts/retrain_er_n40_big_diam7.sbatch` — moved partition `long_gpuh200` → `gpuh200`, raised `--cpus-per-task` 24 → 32, `--num_workers` 24 → 32, time `24:00:00` → `23:55:00`.
- Commit `a386459` on `main` (already pushed).

## What we tried that failed / didn't work / had to be undone

- **Python-BFS APSP** during online data gen: made D≤7 take ~3570s/5k step (≈714 ms/step) → no chance of fitting 1M steps in any partition. Replaced with scipy APSP (now ~360s/5k).
- **`--cpus-per-task=60`** for D≤7: rejected by scheduler ("CPU count per node can not be satisfied" — H200 nodes only have 32 cores total).
- **`--cpus-per-task=32` initial attempts** on `long_gpuh200`: had long queue waits because 32 CPUs = node-exclusive (no other job can share the node's second GPU).
- **`--time=48:00:00` on gpuh200**: rejected (gpuh200 partition max is 24h; long_gpuh200 has 72h).
- **`pip install scipy`** had to be run manually inside the `graph_tf` conda env — scipy isn't pinned in `requirements.txt` (TODO).
- **OOD dependency on a completed job ID**: `--dependency=afterok:<completed_id>` made SLURM refuse with "DependencyNeverSatisfied". Solved by dropping the completed job from the dep list (OOD auto-detects its checkpoint anyway).
- **Hedging strategy executed but cleanup skipped**: 495033 (D≤7 on long_gpuh200, 24 CPU) was the hedge; 495199 (D≤7 on gpuh200, 32 CPU) won the race. We never scancelled 495033, so as of last check both were still running. The plan was to scancel the loser; **must verify and scancel 495033 if still running**, otherwise the OOD's `ls -1dt` will pick the wrong (incomplete) checkpoint when it triggers.

## Instructions the user gave (durable preferences from this session)

- Communicate in Italian unless the technical terms are English.
- Keep responses concise; tabular comparisons are welcome when comparing setups/timings.
- The user is on the Bocconi HPC (account `3352759`), conda env is `graph_tf`. Workflow: edit locally → commit/push → `git pull` on HPC → submit sbatch.
- Never run destructive git ops without asking; `--cpus-per-task` for H200 jobs should stay ≤ 32.
- **Never run `git commit` or `git push` directly.** Edit files locally and write the git commands for the user to run. No Claude signature on commits.
- The user prefers to **hedge** uncertain SLURM submissions (run duplicates on different partitions, scancel the loser) when wait times are uncertain.
- The professor is informal/chill — emails/updates can be in informal Italian without over-formalizing.

## Pending / next steps

1. **Verify 495199 (D≤7, gpuh200, 32 CPU) finished.** Expected completion ~5h after the last status check (was at step 740k/1M with ~360s/5k). If still running, wait.
2. **Scancel 495033 if still running** (D≤7 on long_gpuh200, the hedge) — important to prevent it from updating its `best.pt` and confusing the OOD checkpoint auto-detect.
3. **Verify OOD job 495201** triggered after 495199 completed. If it picked the wrong checkpoint (the 495033 one) because 495033 was still writing when 495201 ran, re-submit OOD with explicit `--checkpoint` paths or after `touch`-ing the 495199 `best.pt`.
4. When OOD finishes: pull the new `runs/report2/ood_eval_n40_big_<JOBID>/` results.
5. **Add `scipy` to `requirements.txt`** (currently must be `pip install`-ed manually).
6. **Update `main.tex`** chapter 5 with the BIG results (in-distribution + OOD) for all four diameter variants + curriculum. Subsubsection structure already in place; only the numbers and 2 mini-comments per variant remain.
7. Send report to prof this weekend; meeting early next week.

## Key job IDs to track (as of session end)

| Job ID | What | Partition | Status at last check |
|--------|------|-----------|----------------------|
| 495198 | BIG D≤9 | gpuh200 | COMPLETED (7h 28m) |
| 495199 | BIG D≤7 (winner) | gpuh200, 32 CPU | RUNNING 15h 17m, step 750k/1M — should finish soon |
| 495033 | BIG D≤7 (hedge) | long_gpuh200, 24 CPU | RUNNING, step ~570k — **SCANCEL when verified 495199 is done** |
| 495201 | OOD eval | gpuh200 | PENDING (Dependency on 495199) |

## Useful commands for resuming

```bash
# Check user's queue
squeue -u 3352759

# Inspect a running training's last logged step
tail -n 5 out/big_er_n40_diam7_495199.out

# Get wall-clock for completed jobs
sacct -j <ID1>,<ID2> --format=JobID,JobName%30,Partition,AllocCPUS,ReqMem,Elapsed,MaxRSS,State

# After OOD completes, see results
ls runs/report2/ood_eval_n40_big_*/
```
