# Time-lapse elastic-wave FWI with baseline regularization

This repository implements **2D elastic full-waveform inversion** using an **RNN + finite-difference (FD)** propagator: differentiable elastic forward modeling with `rnn2D` (Vp, Vs, and density ρ, with coupled stress–velocity updates), direct optimization of those parameter grids, and **time-lapse–style** regularization via MSE against fixed **baseline** models. The main driver fits horizontal and vertical receiver components (accelerometer line data); data I/O and scripts target the Cami line processing pipeline.


## Repository layout

| File | Role |
|------|------|
| `Main_AC_only_bm.py` | **Main entry**: reads sweep, line data, initial and baseline velocity models; builds `FWI2D` and runs training; writes outputs under `results_dir`. |
| `Main_AC_only_bm.ipynb` | Jupyter equivalent of the script (includes optional module reload boilerplate). |
| `C_FWI_V_1_for_Cami_time_lapes_baseline_AC_DAS.py` | **`FWI2D`**: `DataLoader`, training loop, `train_one_epoch`, loss assembly, and `forward_process` calling `rnn2D`. |
| `rnn_fd_elastic2_1D_kernel_DAS.py` | **`rnn2D` / `Propagator2D`**: single-step **elastic** FD recursion (Vp, Vs, ρ), multi-shot batches; see file header for GEOPHYSICS citation. |
| `RNN_FWI_objective_function.py` | **`FWI_costfunction`**: `obj_option` selects L1, L2, correlation-style, or related objectives. |
| `Normalization.py` | **`Nromalization_records_min_max`**: trace normalization by `normalization_method` (observed data and synthetics may use different settings). |
| `generate_source_Cami.py` | **`Reading_Cami_sweep`**: loads `Org_sweep.mat`, autocorrelation, resampling, Butterworth, minimum phase, etc.; outputs `wavelet`. |
| `generate_data_Cami_AC.py` | **`Reading_Cami_data_AC`**: filters horizontal/vertical accelerometer gathers, mute, normalization; yields `shots_obs_x` / `shots_obs_z`. |
| `generate_DAS_1Cdata_Cami.py` | **`Reading_Cami_data_DAS`**: DAS line reader (the baseline main script may pass zero placeholders for DAS). |
| `FWI_filter.py` | **`Filter_Butter`**: Butterworth filtering on shot gathers. |
| `Filter_source.py` | Source wavelet filtering utilities (used by `Reading_Cami_sweep`). |
| `plotimagesc.py` | Plotting helpers: `imagesc`, `add_colorbar`, etc. |
| `H_V_Smooth.py` | Horizontal smoothing utilities (e.g. `H_Smooth` in `Seeing_results.ipynb` / `Seeing_tlresults.ipynb`). |

## Required data files

- `./Org_sweep.mat` — vibroseis sweep (`Reading_Cami_sweep`)
- `./line4/line4_accel_V3.mat` — accelerometer line data and mute arrays (e.g. `monitorV_ns_nt_nr`, `monitorH_ns_nt_nr`, `mute_dh5`, `offset`, `Depthdh`)
- `./vpvsrho_fit_coeff.mat` — Vp–Vs–ρ relationship coefficients in `coeff` (`loadmat` inside `FWI2D.train`)
- `./line4/results/.../vmodel*.pt` — **initial** and **baseline** model lists from a prior inversion step (`torch.load` picks specific frames in `Main_AC_only_bm.py`)

Training writes `train_loss_history.pt`, `vmodel*_list.pt`, gradient lists, and predicted gathers under `results_dir` (e.g. `./line4/results/`).

## How to run

From the project root:

```bash
python Main_AC_only_bm.py
```

Or open `Main_AC_only_bm.ipynb` and run cells in order.

## `FWI2D` loss terms (current code)

In `train_one_epoch`, each batch loss is composed approximately of:

- **Data misfit**: horizontal + vertical synthetics vs. observations via `FWI_costfunction` (L1 / L2 / correlation per `obj_option`).
- **Vp–Vs rock-physics link**: mean squared `(vs_coeff - vmodel2)^2`, where `vs_coeff` is a polynomial in Vp using `vpvsrho_fit_coeff.mat`.
- **Horizontal TV-style regularizer**: mean of `|vp[:,:,:-1] - vp[:,:,1:]|`, weighted by `vp_hor_decay`.
- **Baseline deviation MSE**: `lambda1 * MSE(Vp - Vp_bs) + lambda2 * MSE(Vs - Vs_bs) + lambda3 * MSE(ρ - ρ_bs)`.

Gradients are masked by `mask_grad`, clipped with `clip_grad_norm_`, and applied with Adam to `vmodel1/2/3`; `vmodel*` values are `clamp_`d to physical ranges inside `torch.no_grad()`.




