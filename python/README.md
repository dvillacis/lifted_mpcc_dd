# Plotting helpers

Python scripts that render figures from the solver's output. The solving is done
by `tv_dd` in [`../cpp`](../cpp); these scripts need only numpy and matplotlib.

| file | what it is |
|------|------------|
| `plot_slurm.py`   | seven panels per solution (noisy, recon, diff, delta, index sets, complementarity residual, continuation) from a directory of `tv_dd --save-solution` files |
| `plot_style.py`   | shared figure style: `screen` (200-dpi PNG) and `paper` (two-column PDF) presets, Okabe–Ito palette, figure sizing helpers |
| `plot_discretization.py` | the grid, tiles and consensus copies of the discretization |
| `plot_dataset.py`, `aggregate_runs.py` | figures and tables for dataset runs and sweep timings of the archived drivers; kept because they reuse `plot_slurm.py`'s decoder |

The data-generation helpers of the archived drivers (`dump_data*.py`,
`mpcc_utils.py`) are in [`../archive/python`](../archive/python).

## Setup

Managed with [uv](https://docs.astral.sh/uv/). From this directory:

```bash
uv sync          # creates .venv with the locked versions (uv.lock)
```

`uv` reads `.python-version` and fetches CPython 3.11 itself if needed. Without
uv, `pip install -r requirements.txt` into a 3.11 venv installs the same versions.

## Plotting a run

`plot_slurm.py` takes a directory containing a `sols/` subfolder:

```bash
cd ../cpp
OMP_NUM_THREADS=1 mpirun -np 8 ./tv_dd --data ../images/mariposa.png --size 256 --nsub 8 \
    --save-solution ../runs/sols/sol_mariposa_256_N8.npz
cd ../python
uv run python plot_slurm.py ../runs      # -> ../runs/plots/mariposa_256_N8_*.png
```

`--only TAG` restricts to one solution, `--diff residual` switches the difference
panel to `u−f`, `--out DIR` redirects. The `.npz` is self-contained (the instance,
the solution, a run summary and the μ-trace), so nothing else is needed.
