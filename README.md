# PHASE: corrected implementation and experiments

This folder contains the corrected code behind the results in `../SPL_submission/main.tex`. The original code in `../Code/` is unchanged.

## Problems found in the original SEA experiment (`Code/other_datasets/ensemble_model_sea_dataset.py`)

| # | Problem | Consequence |
|---|---------|-------------|
| 1 | `load_sea_dataset` reassigns `dataset` inside `for ... in enumerate(dataset)`. Python keeps consuming the original iterator. | All 50,000 samples come from one concept (θ = 8). The stream had **no drift**; every segment's class share is ≈35%. |
| 2 | `detect_drift` updates the running mean with the global index `t+1`, not the time since the last reset. Its input is unscaled features (cubes reach ~1000). | `‖x − mean‖ − δ` is always positive, so the statistic grows linearly. It fires at **every multiple of `min_segment_size`** (5000, 10000, …), whatever the data. |
| 3 | A feature-only monitor cannot see SEA drift, because SEA changes only P(y\|x) while P(x) stays uniform. | Phase I cannot work on SEA even after fixing #2. |
| 4 | Class costs were hard-coded to `{0: 2.0, 1: 0.7}`. The weighted/unweighted choice used plain accuracy. | Every segment picked unweighted DT and LR, so the costs were **never used**. |
| 5 | Base-learner posteriors for the weights and the meta-learner were in-sample. | The depth-10 tree is over-confident, which inflates its weight and leaks into stacking. |
| 6 | Engineered features used whole-stream statistics (`qcut`, median, rank, z-score over train **and test**). The "decision" indicators used thresholds of about 1, while the features range over [0, 10]. | Look-ahead leakage, plus near-constant features that were forced into the selection. |
| 7 | `select_best_features` appends the must-include features, then keeps the first 60 indices **in column order**. | High-scoring features were dropped at random. |
| 8 | AUC was computed from hard labels (`roc_auc_score(y_true, y_pred)`). | This equals balanced accuracy, not AUC. |
| 9 | The paper's 86.04% (run `20250614_182336`) and its confusion matrix (run `20250615_083847`, 86.29%) came from different runs. | The reported numbers were inconsistent. |

## What this version does

The same four phases are kept. The fixes:

- **Phase I** (`detector.py`) uses residual-based Page–Hinkley segmentation:
  - At the start of every segment, a class-balanced random forest is fitted on `W_ref = 1000` samples and frozen. Its error indicators are i.i.d. while the concept is unchanged.
  - One PH statistic runs **per class** (a bank of K tests). This way, changes that affect only the minority class are not drowned out by the majority class.
  - After an alarm, the change point is localised by the least-squares single-change estimator, and detection restarts from there.
  - The PH settings `δ = 0.02`, `λ = 25` were chosen on SEA calibration seeds 100–102. Evaluation uses seeds 0–4.
- **Phase II** (`phase.py`):
  - Costs follow Eq. (6), w_c = N_s/(K n_{s,c}).
  - The cost on/off choice per learner uses **cross-fitted balanced accuracy** (5 folds).
  - k-means posteriors are cost-weighted, Laplace-smoothed cluster histograms with k = 10.
- **Phase III**:
  - Weights use the cost-weighted likelihood of the true labels on **out-of-fold** posteriors.
  - The SVM (RBF, C = 1) is admitted through the nested weight α when the fused balanced accuracy is below τ = 0.9. Its posteriors are Platt-calibrated on out-of-fold decision values. It is fitted on at most 10,000 samples per segment.
- **Phase IV**: one random-forest meta-learner (500 trees, depth ≤ 20, balanced classes) is trained on the out-of-fold weighted posteriors of **all** segments, as in the original code.
  - A **stacking gate** keeps the meta-learner only if it beats `argmax p` in cross-validated balanced accuracy on the last segment, using training data only.
  - **Disclosure**: the gate was added *after* the first full run showed that always-on stacking hurt on SEA-5% and Elec2.
  - Both the "always stack" and "never stack" variants are therefore reported in the ablation table. Their runs are kept as `PHASE-alwaysmeta` and `PHASE-nometa`.
- **Inference**: test queries are routed to the base layer of the most recent segment.
- **No hand-crafted features**: every method sees the same raw inputs.

## Datasets

| Key | Stream | n | Classes | Drift |
|-----|--------|---|---------|-------|
| `sea` | SEA, θ: 8 → 9 → 7 → 9.5 at 10k/20k/30k, 10% label noise, natural priors | 50,000 | 2 | known |
| `sea10` | Same, with the class-0 share fixed at 10% by rejection sampling | 50,000 | 2 | known |
| `sea5` | Same, with the class-0 share fixed at 5% | 50,000 | 2 | known |
| `elec2` | Elec2 via River (the `date` column is dropped) | 45,312 | 2 | real |
| `covtype` | Forest Covertype, `../Code/Archive_Drift Analysis/covtype.csv` | 581,012 | 7 | real |

## Protocol

- Temporal hold-out: the first 80% of the stream is for training and the last 20% for testing.
- Every method is frozen after training and sees no test labels.
- SEA has 5 seeds, which vary both the data and the models. Elec2 has 5 model seeds. Covertype has 1 seed.
- Baselines (`baselines.py`):
  - HAT, ARF and SRP from River, with 10 members each.
  - OOB and UOB (Wang, Minku & Yao, TKDE 2015), with 10 Hoeffding trees and decay θ = 0.9.

## Reproduce

```bash
pip install -r requirements.txt      # Python 3.13; Elec2 is downloaded by River (~0.7 MB)
python run_experiments.py --datasets sea sea10 sea5 elec2 --seeds 0 1 2 3 4 --jobs 10
python run_experiments.py --datasets covtype --seeds 0 --jobs 6
python make_tables.py                # ../SPL_submission/tables/*.tex (tables + number macros used in the text)
python summarize.py                  # results/summary.csv, detection.csv, tables.tex
python make_figures.py               # figures/fig_results.pdf (also copied into the paper)
```

The SEA/Elec2 grid takes about 45 minutes on 10 cores. Each Covertype run takes 10–60 minutes.

## Files

- `data.py`: stream loaders, including the corrected SEA generator.
- `detector.py`: Phase I, i.e. the PH bank, least-squares localisation and detection scoring.
- `phase.py`: Phases II–IV and inference.
- `baselines.py`: River baselines, plus the OOB/UOB implementation.
- `metrics.py`: accuracy, balanced accuracy, G-mean, κ, MCC, macro-F1 and per-class recall.
- `run_experiments.py`: the experiment grid. It is resumable and writes one JSON per run to `results/runs/`.
- `summarize.py` and `make_figures.py`: build the paper's tables and figures.

## Results (balanced accuracy %, test window; mean over 5 seeds, Covertype 1 run)

| Method | SEA | SEA-10% | SEA-5% | Elec2 | Covertype |
|---|---|---|---|---|---|
| HAT | 86.3 | 65.6 | 50.5 | 75.7 | 20.1 |
| ARF | 88.6 | 67.6 | 53.5 | 76.0 | 18.4 |
| SRP | 80.7 | 51.1 | 50.1 | 75.5 | 22.7 |
| OOB | 85.2 | 81.5 | 76.4 | 69.8 | **48.9** |
| UOB | 86.9 | 84.5 | 86.1 | 64.6 | 21.0 |
| **PHASE** | **89.4** | **88.8** | **88.3** | **76.9** | 21.9 |

- **Covertype**: PHASE trails OOB. Its last training segment contains only 3 of the 6 classes in the test window. Without segmentation, the same ensemble scores 72.2.
- **Detection on SEA** (15 changes per imbalance level):
  - The class-conditional bank finds 15 / 13 / 10 changes.
  - A pooled PH test finds 15 / 5 / 5. Under imbalance it misses every minority-only change.
- **Archived runs**:
  - `results/runs_archive_alwaysstack/` holds the first ablation runs, made before the gate.
  - `results/runs_stale_covtype_alwaysstack/` holds Covertype runs that orphaned worker processes completed with pre-gate code. They are not used.
