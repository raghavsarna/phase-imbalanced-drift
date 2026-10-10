# Class-conditional Page–Hinkley detection of minority-only drift

Code and experiment logs for the letter *"Class-Conditional Page–Hinkley Detection of Minority-Only Drift in Imbalanced Data Streams"* by Raghav Sarna and Gittaly Dhingra (Plaksha University).

When a concept change affects only a rare class, a drift detector that watches a classifier's pooled error barely notices it, because the change is scaled by the class prior. This repository implements and evaluates two things:

1. **A class-conditional detector.** One Page–Hinkley (PH) test per class monitors the residuals of a frozen reference model. Each alarm is localised by the least-squares single-change estimator. The letter proves two results:
   - The expected delay is at most `(λ+1) / (π_c (Δ_c − δ))`, i.e. inversely proportional to the share `π_c` of the changed class.
   - A pooled test loses its positive drift once `π_c Δ_c ≤ δ`.

   It also bounds the bank's false-alarm probability by a prior-weighted average of per-class terms.
2. **PHASE** (Page–Hinkley Adaptive Segmented Ensemble). A cost-sensitive, likelihood-weighted ensemble of a decision tree, logistic regression and k-means posteriors, fitted to the current segment. An RBF-SVM is added when needed, and a class memory keeps classes that are absent from the latest segment predictable.

## Installation

Python 3.13 is the tested version.

```bash
git clone https://github.com/raghavsarna/phase-imbalanced-drift.git
cd phase-imbalanced-drift
pip install -r requirements.txt
```

The ROSE baseline runs the authors' Java code in MOA. It needs a JDK (tested with OpenJDK 27):

```bash
rose/build.sh      # fetches MOA (capymoa 0.15.1 wheel) and ROSE (commit c86c8e1), compiles rose/RunROSE.java
```

## Datasets

| Key | Stream | Samples | Classes | Source |
|---|---|---|---|---|
| `sea`, `sea20`, `sea10`, `sea5`, `sea2`, `sea1` | SEA, θ: 8→9→7→9.5 at 10k/20k/30k, 10% label noise. The class-0 share is natural or fixed at 20/10/5/2/1% by rejection sampling. | 50,000 | 2 | River (generated) |
| `seanull`, `seanull5` | Drift-free SEA (θ = 8), natural and 5% minority | 50,000 | 2 | River (generated) |
| `elec2` | Electricity prices (`date` dropped) | 45,312 | 2 | River download |
| `covtype` | Forest Covertype, standard order | 581,012 | 7 | `data/covtype.csv` or `$PHASE_COVTYPE_CSV` |
| `insects_abrupt`, `insects_gradual` | INSECTS abrupt/gradual imbalanced (Souza et al., 2020) | 355,275 / 143,323 | 6 | River download |
| `creditcard` | Credit-card fraud, 0.17% fraud (`Time` dropped) | 284,807 | 2 | River download |

## Usage

```python
from phase import PHASE, PhaseConfig

model = PHASE(PhaseConfig(seed=0), n_classes=K).fit(X_train, y_train)   # X: (n, m) in stream order, y: ints 0..K-1
y_pred = model.predict(X_test)
print(model.boundaries)        # estimated change points

from detector import segment   # the detector alone
boundaries, alarms = segment(X, y, K, mode="class", test="ph")   # mode: class | pooled, test: ph | ddm | adwin
```

### Reproduce the letter

```bash
python theory_sim.py                                  # Monte Carlo check of Props. 1-2 -> results/theory.json
python detection_study.py --jobs 4                    # six detectors on SEA -> results/detection/
python run_experiments.py --protocol holdout --jobs 6 # 8 streams x 19 methods x 5 seeds -> results/holdout/
python run_experiments.py --protocol preq --jobs 6    # prequential -> results/preq/
python run_experiments.py --protocol holdout --datasets sea20 sea2 sea1 \
    --methods PHASE PHASE-pooled PHASE-noseg HAT ARF SRP OOB UOB ARF-US ROSE   # minority-share sweep (Fig. 2(b))
python timeline.py --datasets sea10                   # per-sample prequential predictions (Fig. 2(a)) -> results/timeline/
python summarize.py                                   # overview, CSV summaries
python make_tables.py                                 # LaTeX tables, number macros, Friedman/Nemenyi/Wilcoxon -> results/stats.json
python make_figures.py                                # Figs. 1 and 2
```

- Every run writes one JSON file and is skipped if that file already exists, so the scripts can be resumed.
- Detector outputs are cached in `results/cache/`.
- On 12 cores the full grid takes roughly 8–10 hours. SRP is the slowest learner, at up to an hour per large-stream run.
- `make_tables.py` and `make_figures.py` write into the paper folder (`$PHASE_PAPER_DIR`, or `../SPL_submission` if it exists), otherwise into `results/latex/`.

## Protocols and settings

- **Hold-out** (deployment without labels). Train on the first 80% of the stream (stream learners make one pass), freeze, and predict the last 20%.
- **Prequential** (test-then-train). Predict each sample, then learn from it. The first 1000 predictions are not scored.
  - PHASE runs its detector online.
  - It refits on the current segment (at most the last `W_max` = 10,000 samples, plus the class memory) every `R` = 1000 samples and after every alarm.
- **Seeds.** 5 per setting: data and model seeds on SEA, model seeds on the real streams.
- **Baselines.**
  - HAT, ARF and SRP from River, with 10 members.
  - OOB/UOB (Wang et al., 2015), with 10 Hoeffding trees and decay 0.9.
  - ARF-US: ARF trained on UOB's undersampled stream.
  - ROSE (Cano & Krawczyk, 2022) with default options.
- **PHASE settings.**
  - Reference model: a class-balanced 100-tree random forest, with `W_ref = L_min = 1000`.
  - `δ = 0.02` and `λ = 25`, calibrated on SEA seeds 100–102, which are not used for evaluation.
  - Class memory `B = 100`, `F = 5` folds, `τ = 0.9`, and the SVM on at most 10,000 samples.
- **Detectors compared.** {class-wise, pooled} × {PH, DDM, ADWIN}, all on the same residuals and with the same localisation. DDM and ADWIN use River defaults.

## Results (balanced accuracy %, mean of 5 seeds)

| Method | SEA | SEA-10% | SEA-5% | Elec2 | Covertype | INSECTS-A | INSECTS-G | CreditCard | Avg. rank |
|---|---|---|---|---|---|---|---|---|---|
| *Hold-out* | | | | | | | | | |
| ROSE | 88.8 | 84.0 | 75.0 | 77.7 | 28.6 | 60.7 | 55.8 | 84.5 | 3.12 |
| ARF-US | 88.6 | 86.4 | 85.9 | 70.4 | 28.7 | 55.7 | 54.6 | **92.3** | 3.12 |
| OOB | 85.2 | 81.5 | 76.4 | 69.8 | **50.7** | 51.6 | 53.2 | 79.7 | 4.88 |
| **PHASE** | **88.9** | **88.4** | **88.3** | **77.7** | 37.7 | **67.1** | **59.1** | 91.6 | **1.25** |
| *Prequential* | | | | | | | | | |
| ROSE | 87.1 | 82.5 | 77.6 | **88.7** | **86.8** | 68.7 | 67.1 | 88.3 | 2.62 |
| ARF-US | 86.0 | 84.1 | 83.3 | 83.7 | 32.1 | 59.4 | 57.1 | **91.9** | 3.88 |
| **PHASE** | **87.2** | **85.8** | **84.7** | 76.4 | 80.6 | **69.9** | **68.6** | 91.2 | **2.25** |

All seven baselines are in `results/stats.json` and in Table I of the letter. Ranks are computed over all eight methods.

- **Detection on SEA** (15 changes per minority share of 20 / 10 / 5%):
  - The class-wise PH bank detects 15 / 13 / 10; pooled PH detects 7 / 5 / 5.
  - Class conditioning lifts ADWIN from 20 to 37 detections and DDM from 15 to 35. Class-wise DDM, however, raises many false alarms.
  - No detector is reliable at 2% or 1%.
- **Minority-share sweep (hold-out, Fig. 2(b)).** PHASE has the highest BA from the natural share down to 5%. At 2% and 1% few or no changes are detected, and PHASE (82.8 / 82.1) falls below UOB (86.2 / 83.7) and ARF-US (84.0 / 83.3).
- **Recovery after a minority-only change (Fig. 2(a)).** On SEA-10%, over the 5,000 samples after the change θ: 7 → 9.5, prequential BA is:
  - PHASE: 83.0%
  - PHASE with a pooled detector: 77.6%
  - PHASE without a detector: 77.7%
  - ROSE: 73.4%
- **Statistics.**
  - Hold-out: Friedman p < 0.001. PHASE beats 4 baselines by more than the Nemenyi critical difference (3.71).
  - With eight streams, the smallest attainable Holm-adjusted Wilcoxon p-value is 0.055.
  - Prequential: Friedman p = 0.0014; PHASE has the best rank, but only HAT differs from it by more than the critical difference.
- **Limitations.**
  - Covertype's spatial order makes classes appear and vanish, so training on the latest segment loses to OOB in the hold-out protocol. The class memory recovers 14.5 of the 40 points lost.
  - In prequential mode, learners that update after every sample win on the strongly autocorrelated Elec2 and on Covertype.

## Repository layout

```
data.py              stream loaders
detector.py          class-wise / pooled PH, DDM, ADWIN on reference-model residuals; LS localisation
phase.py             PHASE (costs, base learners, likelihood weights, SVM refinement, class memory)
prequential.py       test-then-train evaluation of PHASE and of the stream baselines
baselines.py         River baselines, OOB/UOB, ARF-US, ROSE runner
rose/                Java runner for ROSE (RunROSE.java) and build script
theory_sim.py        Monte Carlo check of the delay and false-alarm bounds
detection_study.py   detector comparison on SEA
timeline.py          per-sample prequential predictions for the accuracy-over-time plot
run_experiments.py   experiment grid (protocol x dataset x method x seed)
summarize.py, make_tables.py, make_figures.py   analysis, LaTeX tables, statistics, Figs. 1-2
results/holdout/, results/preq/, results/detection/   one JSON per run
results/timeline/    per-sample predictions (npz) for Fig. 2(a)
results/archive_v1/  logs of an earlier configuration of the method (stacked meta-learner, three datasets), kept for reference
```

## Citation

```
R. Sarna and G. Dhingra, "Class-Conditional Page–Hinkley Detection of Minority-Only Drift
in Imbalanced Data Streams," submitted to IEEE Signal Processing Letters.
```
