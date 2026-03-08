# Reproduction Report: dynamax

**Repository:** probml/dynamax
**Date:** 2026-03-08
**Reproducer:** Automated (Claude Code)

---

## Section 1 — Summary Table

### Test Suite (pytest)

| Asset | Type | Status | Notes |
|-------|------|--------|-------|
| `dynamax/generalized_gaussian_ssm/inference_test.py` | test | PASS | 1/1 passed |
| `dynamax/generalized_gaussian_ssm/models_test.py` | test | PASS | 3/3 passed |
| `dynamax/hidden_markov_model/inference_test.py` | test | PASS | 12/12 passed |
| `dynamax/hidden_markov_model/models/test_models.py` | test | PASS | 19/19 passed |
| `dynamax/linear_gaussian_ssm/inference_test.py` | test | PASS | 3/3 passed |
| `dynamax/linear_gaussian_ssm/info_inference_test.py` | test | PASS | 7/7 passed |
| `dynamax/linear_gaussian_ssm/models_test.py` | test | PASS | 3/3 passed |
| `dynamax/linear_gaussian_ssm/parallel_inference_test.py` | test | PASS | 15/15 passed |
| `dynamax/nonlinear_gaussian_ssm/inference_ekf_test.py` | test | PASS | 5/5 passed |
| `dynamax/nonlinear_gaussian_ssm/inference_ukf_test.py` | test | PASS | 1/1 passed |
| `dynamax/parameters_test.py` | test | PASS | 4/4 passed |
| `dynamax/slds/inference_test.py` | test | PASS | 2/2 passed |
| `dynamax/utils/distributions_test.py` | test | PASS | 14/14 passed |
| `dynamax/utils/test_optimize.py` | test | PASS | 1/1 passed |
| `dynamax/utils/utils_test.py` | test | PASS | 4/4 passed |

**Total: 95/95 tests passed** (276s, 389 warnings)

### Jupyter Notebooks (docs/notebooks)

| Asset | Type | Status | Notes |
|-------|------|--------|-------|
| `docs/notebooks/hmm/gaussian_hmm.ipynb` | notebook | PASS | |
| `docs/notebooks/hmm/casino_hmm_inference.ipynb` | notebook | PASS | |
| `docs/notebooks/hmm/casino_hmm_learning.ipynb` | notebook | PASS | |
| `docs/notebooks/hmm/autoregressive_hmm.ipynb` | notebook | PASS | |
| `docs/notebooks/hmm/custom_hmm.ipynb` | notebook | PASS | |
| `docs/notebooks/linear_gaussian_ssm/kf_tracking.ipynb` | notebook | PASS | |
| `docs/notebooks/linear_gaussian_ssm/kf_linreg.ipynb` | notebook | PASS | |
| `docs/notebooks/linear_gaussian_ssm/lgssm_learning.ipynb` | notebook | PASS | |
| `docs/notebooks/linear_gaussian_ssm/lgssm_hmc.ipynb` | notebook | PASS | |
| `docs/notebooks/linear_gaussian_ssm/lgssm_parallel_inference.ipynb` | notebook | PASS | |
| `docs/notebooks/nonlinear_gaussian_ssm/ekf_ukf_pendulum.ipynb` | notebook | PASS | |
| `docs/notebooks/nonlinear_gaussian_ssm/ekf_ukf_spiral.ipynb` | notebook | PASS | |
| `docs/notebooks/nonlinear_gaussian_ssm/ekf_mlp.ipynb` | notebook | PASS | |
| `docs/notebooks/generalized_gaussian_ssm/cmgf_poisson_demo.ipynb` | notebook | PASS | |
| `docs/notebooks/generalized_gaussian_ssm/cmgf_logistic_regression_demo.ipynb` | notebook | PASS | |
| `docs/notebooks/generalized_gaussian_ssm/cmgf_mlp_classification_demo.ipynb` | notebook | PASS | |
| `docs/notebooks/slds/rbpf_maneuver.ipynb` | notebook | FIXED | See Patches #1 and #2 |

**Total: 17/17 docs notebooks passed** (2 required patches)

---

## Section 2 — Environment Specification

- **OS:** macOS Darwin 25.3.0 arm64 (Apple Silicon)
- **Python:** 3.11.14 (conda env: `dynamax`)
- **JAX:** 0.9.1
- **jaxlib:** 0.9.1
- **CUDA:** N/A (CPU-only, Apple Silicon)
- **dynamax version:** 1.0.1+11.g3f4366b (editable install from source)

Full `pip freeze` captured at `/tmp/dynamax_pip_freeze.txt` (159 packages).

Key dependencies installed:
```
jax==0.9.1
jaxlib==0.9.1
tfp-nightly==0.26.0.dev20260307
optax==0.2.6
scikit-learn==1.8.0
matplotlib==3.10.8
flax==0.12.5
blackjax==1.3
numpy==2.4.2
scipy==1.17.1
```

---

## Section 3 — Patch Log

### Patch #1: `docs/notebooks/slds/rbpf_maneuver.ipynb` (Cell 7)

**File:** `docs/notebooks/slds/rbpf_maneuver.ipynb`
**Original:** `ohe = OneHotEncoder(sparse=False)`
**Patched:** `ohe = OneHotEncoder(sparse_output=False)`
**Reason:** scikit-learn >= 1.2 renamed the `sparse` parameter to `sparse_output`. The old name was removed in scikit-learn 1.4+.

### Patch #2: `docs/notebooks/slds/rbpf_maneuver.ipynb` (Cell 7)

**File:** `docs/notebooks/slds/rbpf_maneuver.ipynb`
**Original:**
```python
ax.w_xaxis.set_pane_color((0, 0, 0, 0))
ax.w_yaxis.set_pane_color((0, 0, 0, 0))
ax.w_zaxis.set_pane_color((0, 0, 0, 0))
```
**Patched:**
```python
ax.xaxis.set_pane_color((0, 0, 0, 0))
ax.yaxis.set_pane_color((0, 0, 0, 0))
ax.zaxis.set_pane_color((0, 0, 0, 0))
```
**Reason:** matplotlib >= 3.8 removed the `w_xaxis`/`w_yaxis`/`w_zaxis` aliases from `Axes3D`. The correct attributes are `xaxis`/`yaxis`/`zaxis`.

---

## Section 4 — Data Manifest

No external datasets required. All demos and tests generate synthetic data at runtime.

---

## Section 5 — Verdict

### FULLY REPRODUCED

All 95 tests pass. All 17 docs notebooks execute successfully (2 required minor API compatibility patches for scikit-learn and matplotlib version updates). No external data dependencies. No GPU required. The library is fully functional on Python 3.11 with latest JAX/jaxlib 0.9.1.
