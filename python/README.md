# Python reference & reproduction helpers

This directory holds the **numerical reference** for the lifted TV-denoising MPCC
and the small scripts that feed / visualize the C++ domain-decomposition solver
in [`../cpp`](../cpp). Everything the C++ side is validated against lives here.

## Contents

**Reference implementations** (the monolithic IPOPT/`cyipopt` solvers — the
ground truth the C++ DD plugin reproduces):

| file | what it is |
|------|------------|
| `lifted_mpcc_unitball_v2.py`        | numerics-hardened square-grid solver (the base reference; defines `load_image`, `HSLLIB`, `psnr`) |
| `lifted_mpcc_unitball_staggered.py` | staggered (cell-centred) discretization |
| `lifted_mpcc_1d.py`                 | staggered **1D** grid + domain-decomposition probe |
| `lifted_mpcc_2d.py`                 | staggered **2D** grid + domain-decomposition probe |

**Helpers** (glue between Python and the C++ solver):

| file | what it does |
|------|--------------|
| `dump_data.py`, `dump_data_1d.py`, `dump_data_2d.py` | write a solver instance (clean + noisy image, CP warm start, owner map) to a `.txt` the C++ drivers load — the only route where Python and C++ solve a **byte-identical** instance |
| `plot_1d.py`, `plot_2d.py` | draw the solution + arrowhead figures from the C++ solver's `--save-solution` / `--save-dd` dumps |
| `plot_slurm.py`            | publication figures from a SLURM result directory |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11
pip install -r requirements.txt
```

`cyipopt` links against a system IPOPT (Homebrew `ipopt` / conda-forge `ipopt`);
see [`../README.md`](../README.md) for the toolchain.

## Two ways to use it

**1 — Run the reference solver directly** (no C++ needed):

```bash
python lifted_mpcc_unitball_v2.py                 # cameraman N=32 σ=0.1
python lifted_mpcc_2d.py --nsub 4                 # staggered 2D + DD probe
```

The bundled test image is `images/cameraman.png` beside these scripts (the
scripts' `--data` default resolves there). Pass `--data <path>` for another
image.

**2 — Dump an instance for the C++ DD solver, then plot its output:**

```bash
# generate a byte-identical instance for cpp/dd_solve_1d
python dump_data_1d.py --n 64 --nsub 4 -o ../cpp/data/data_1d_64.txt

# ... solve it in C++ (see ../cpp/README.md) with --save-solution / --save-dd,
#     then render the figures:
python plot_1d.py --data ../cpp/data/data_1d_64.txt \
    --solution sol_1d_64.txt --dd-dump dd_1d_64.txt \
    --save-plot s.png --save-dd-plot dd.png
```

## HSL MA57 (`HSLLIB`)

`lifted_mpcc_unitball_v2.py` picks the linear solver from `--linear-solver`
(default `ma57`). If HSL MA57 is present it is used, otherwise the code prints a
warning and falls back to MUMPS. Point `HSLLIB` at your MA57 shared library:

```bash
export HSLLIB=/path/to/libhsl_ma57.{dylib,so}
```

(The file ships with a developer's default path baked in; the `HSLLIB` env var
overrides it. MA57 is proprietary — see the top-level README for how to obtain
it. MUMPS, bundled with IPOPT, works without it.)
