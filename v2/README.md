# Label-routed CUSUM detection of minority-class drift (revised letter)

Code for the resubmission of SPL-49248-2026, *"Label-Routed CUSUM Detection of Minority-Class Drift in Imbalanced Data Streams"*. The code of the first version (class-routed Page–Hinkley test and the PHASE ensemble) remains in the parent folder. Its logs feed Section S7 of the supplement.

## Files

| File | Content |
|---|---|
| `detectors.py` | Label-routed Bernoulli LR-CUSUM bank (proposed), multi-design bank, pooled LR-CUSUM, risk-adjusted CUSUM, Page–Hinkley, DDM, DDM-OCI, LFR (numba) |
| `baselines.py` | ADWIN (River) and the RBM-IM reimplementation (choices documented in the docstring) |
| `streams.py` | The 11 benchmark streams with a known minority-class change, their drift-free versions, and the prior-shift streams |
| `bench.py` | Matched-ARL benchmark. Oracle quantities, threshold calibration on 30 drift-free streams, then evaluation on 30 streams with a change and 30 without |
| `sim.py` | Monte Carlo study of Propositions 1–3 on Bernoulli residuals (Fig. 1, Tables S3–S5) |
| `robust.py` | Label availability, reference-model quality, and prior shift (Tables S6–S8) |
| `retrain.py` | Detector-triggered retraining with a fixed learner (Table II) |
| `phase_supp.py` | Tables of the first version's PHASE ensemble (Tables S9–S11) |
| `report.py` | Every table, figure, and number macro of the letter, the supplement, and the response |

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install river numpy scipy pandas scikit-learn matplotlib numba
cd v2
../.venv/bin/python sim.py                                   # ~1 h on 12 cores
../.venv/bin/python bench.py --families sea20 sea10 sea5 sea2 sea1 gauss5 gauss1 gmm5 credit covtype insects --jobs 11
../.venv/bin/python robust.py                                # prior, labels, refmodel
for f in sea5 gauss5 gmm5 credit covtype; do ../.venv/bin/python retrain.py $f & done; wait
../.venv/bin/python phase_supp.py
../.venv/bin/python report.py                                # writes ../../SPL_resubmission/{tables,figures}
```

Data:
- CreditCard and INSECTS are downloaded by River into `~/river_data`.
- Covertype is read from `../data/covtype.csv` or from `$PHASE_COVTYPE_CSV`.

Behaviour:
- Every script resumes, because finished runs are kept in `../results_v2/`.
- The first call to LFR builds its Monte Carlo bound tables (`results_v2/lfr_tails*.npz`, about 6 minutes on 12 cores).

## Protocol in brief

- **Reference model and rates.** A class-balanced random forest is fitted on `W_ref` samples and frozen. Per-class residual rates are frozen on the next `W_cal` samples. Monitoring starts at `t0 = W_ref + W_cal`.
- **Calibration.** Each detector's threshold is set by bisection so that its alarm rate on 30 drift-free streams of the family is 1/ARL, with ARL = 5·10⁴.
- **Evaluation.** Delay is the time from the change to the first alarm after it. Statistics restart after every alarm, and misses count as the remaining stream length. There are 30 seeds per stream.
- **Oracle tuning.** The pooled LR-CUSUM and both PH tests receive the true post-change shifts. These are measured on 5 separate streams.
