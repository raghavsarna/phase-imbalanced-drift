# PHASE: change-point-segmented stacked ensembles for imbalanced data streams

This repository contains the code and experiment logs for the letter *"PHASE: Change-Point-Segmented Stacked Ensembles for Class-Imbalanced Non-Stationary Data Streams"* by Raghav Sarna and Gittaly Dhingra (Plaksha University).

**PHASE** (**P**age–**H**inkley-segmented **A**daptive **S**tacked **E**nsemble) learns from a labelled data stream whose distribution changes over time (concept drift) and in which some classes are rare (class imbalance). It has two parts:

1. **Segmentation.** PHASE cuts the stream into pieces where the concept stays the same. A frozen reference model's mistakes are monitored per class by a bank of Page–Hinkley tests, so that changes affecting only a rare class are not drowned out by the majority class.
2. **Modelling.** PHASE fits a cost-sensitive stacked ensemble to every piece.

## Method overview

| Phase | What it does | Code |
|---|---|---|
| I | **Class-conditional Page–Hinkley segmentation.** A class-balanced random forest is fitted on the first `W_ref` samples of a segment and frozen. One PH test per class monitors its error indicators. After an alarm, the change point is localised by the least-squares single-change estimator, and detection restarts from there. | `detector.py` |
| II | **Cost-sensitive base learners**: a decision tree, logistic regression, and k-means used as a posterior model (cost-weighted cluster histograms). Each learner uses class costs w_c = N_s/(K n_c) if that improves its cross-fitted balanced accuracy. | `phase.py` |
| III | **Likelihood-weighted fusion.** Each learner's weight is proportional to the cost-weighted probability it assigns to the true labels (out-of-fold). If the fused balanced accuracy is below τ, an RBF-SVM with Platt-calibrated posteriors is added through a nested convex weight α. | `phase.py` |
| IV | **Shared stacked meta-learner.** One random forest is trained on the out-of-fold weighted posteriors of all segments. A stacking gate keeps it only if it beats the fused rule in cross-validated balanced accuracy on the latest segment, using training data only. | `phase.py` |
| Inference | New samples are predicted by the most recent segment's models. | `phase.py` |

The design rationale and the math, including Proposition 1 (why a pooled test misses minority-only changes), are in the paper.

## Installation

Python 3.13 is the tested version.

```bash
git clone https://github.com/raghavsarna/phase-imbalanced-drift.git
cd phase-imbalanced-drift
pip install -r requirements.txt
```

## Datasets

| Key | Stream | Samples | Classes | Drift |
|-----|--------|---|---------|-------|
| `sea` | SEA (River). θ: 8 → 9 → 7 → 9.5 at 10k/20k/30k, 10% label noise, natural class priors | 50,000 | 2 | known change points |
| `sea10` | SEA, with the class-0 share fixed at 10% by rejection sampling | 50,000 | 2 | known change points |
| `sea5` | SEA, with the class-0 share fixed at 5% | 50,000 | 2 | known change points |
| `elec2` | Elec2 electricity prices (downloaded automatically by River, about 0.7 MB; `date` column dropped) | 45,312 | 2 | real |
| `covtype` | Forest Covertype, in its standard order | 581,012 | 7 | real |

SEA and Elec2 need no manual download. For Covertype, place `covtype.csv` (54 features followed by the label column) at `data/covtype.csv`, or set `PHASE_COVTYPE_CSV=/path/to/covtype.csv`.

## Usage

### Use PHASE on your own stream

```python
from phase import PHASE, PhaseConfig

model = PHASE(PhaseConfig(seed=0), n_classes=K).fit(X_train, y_train)   # X: (n, m) array in stream order, y: ints 0..K-1
y_pred = model.predict(X_test)
print(model.boundaries)   # estimated change points in the training stream
```

### Reproduce the paper

```bash
python run_experiments.py --datasets sea sea10 sea5 elec2 --seeds 0 1 2 3 4 --jobs 10
python run_experiments.py --datasets covtype --seeds 0 --jobs 6
python summarize.py       # results/summary.csv, results/detection.csv
python make_tables.py     # LaTeX tables + number macros used in the paper
python make_figures.py    # figures/fig_results.pdf
```

- `run_experiments.py` is resumable: it writes one JSON per run to `results/runs/` and skips runs that already exist.
- The SEA/Elec2 grid takes about 45 minutes on 10 cores. Each Covertype run takes 10–60 minutes.
- `make_tables.py` and `make_figures.py` write into the paper folder (`$PHASE_PAPER_DIR`, or `../SPL_submission` if it exists). Otherwise they write to `results/latex/`.

## Experimental protocol

- **Temporal hold-out.** Every method trains on the first 80% of the stream; stream learners make a single pass. All models are then frozen and predict the last 20% without labels.
- **Seeds.**
  - SEA: 5 seeds, which vary the data and the models.
  - Elec2: 5 model seeds.
  - Covertype: 1 run.
- **Baselines** (`baselines.py`):
  - HAT, ARF and SRP from River, with 10 members each.
  - OOB and UOB (Wang, Minku & Yao, IEEE TKDE 2015), with 10 Hoeffding trees and decay 0.9.
- **PHASE settings.**
  - `W_ref = L_min = 1000`, with `δ = 0.02` and `λ = 25` chosen on separate SEA calibration seeds (100–102).
  - 5-fold cross-fitting, `τ = 0.9`, SVM capped at 10,000 samples per segment, and k-means with `k = 10`.
  - Meta-learner: a random forest with 500 trees and depth ≤ 20.
- **Metrics.**
  - Main: balanced accuracy and Cohen's κ.
  - Also logged in every run's JSON: accuracy, G-mean, MCC, macro-F1 and per-class recall.

## Results

Test balanced accuracy (%). SEA and Elec2 are averaged over 5 seeds; Covertype is a single run.

| Method | SEA | SEA-10% | SEA-5% | Elec2 | Covertype |
|---|---|---|---|---|---|
| HAT | 86.3 | 65.6 | 50.5 | 75.7 | 20.1 |
| ARF | 88.6 | 67.6 | 53.5 | 76.0 | 18.4 |
| SRP | 80.7 | 51.1 | 50.1 | 75.5 | 22.7 |
| OOB | 85.2 | 81.5 | 76.4 | 69.8 | **48.9** |
| UOB | 86.9 | 84.5 | 86.1 | 64.6 | 21.0 |
| **PHASE** | **89.4** | **88.8** | **88.3** | **76.9** | 21.9 |

- **Change detection on SEA** (15 true changes per imbalance level):
  - The class-conditional PH bank detects 15 / 13 / 10 changes at natural / 10% / 5% minority share.
  - A pooled PH test detects 15 / 5 / 5. Under imbalance it misses every change that only affects the minority class.
- **Limitation (Covertype).** PHASE predicts with the most recent segment. On Covertype that segment contains only 3 of the 6 classes in the test window. Trained without segmentation, the same ensemble reaches 72.2%.
- **Ablations** (`PHASE-*` runs):
  - Segmentation and class costs give the largest gains.
  - The stacking gate tracks whichever of "always stack" and "never stack" is better.
  - The gate was introduced after a first full run showed that always-on stacking hurt on SEA-5% and Elec2. Both variants are kept and reported for transparency.

## Repository layout

```
data.py             stream loaders (SEA with injected drift and controlled imbalance, Elec2, Covertype)
detector.py         Phase I: class-conditional PH bank, least-squares localisation, detection scoring
phase.py            Phases II-IV and inference (PHASE, PhaseConfig)
baselines.py        River baselines and an OOB/UOB implementation
metrics.py          balanced accuracy, G-mean, kappa, MCC, macro-F1, per-class recall
run_experiments.py  experiment grid (methods x datasets x seeds)
summarize.py        aggregate results into CSV summaries
make_tables.py      LaTeX tables and number macros for the paper
make_figures.py     Fig. 2 of the paper
results/runs/       one JSON per run (config, segmentation, per-segment weights, metrics)
results/runs_archive_alwaysstack/      ablation runs made before the stacking gate was introduced
results/runs_stale_covtype_alwaysstack/ superseded Covertype runs from pre-gate code (not used)
figures/            generated figures
```

## Citation

If you use this code, please cite the letter (details to be added on publication):

```
R. Sarna and G. Dhingra, "PHASE: Change-Point-Segmented Stacked Ensembles for
Class-Imbalanced Non-Stationary Data Streams," submitted to IEEE Signal Processing Letters.
```
