# Reproduction helpers (data generation + plotting)

Small Python scripts that (1) build **byte-identical input instances** for the
C++ domain-decomposition solver in [`../cpp`](../cpp), and (2) render figures from
its output. The solving itself is done in C++; these scripts need no IPOPT.

## Contents

| file | what it is |
|------|------------|
| `mpcc_utils.py`   | the generic utility module — the data-generation core extracted from the project's Python reference solver: synthetic/real instance builders (`make_signal`, `make_phantom`, `load_image`), the staggered 1D/2D lifted-MPCC problem objects (`Lifted1DMPCC`, `Lifted2DMPCC`), the Chambolle–Pock warm start (`initial_point`), and the domain-decomposition partition + KKT owner-map rule (`Partition1D/2D`, `kkt_owner`). |
| `dump_data.py`    | write the plain `N + clean + noisy` instance for the **uniform 2D** driver (`dd_solve`) |
| `dump_data_1d.py` | write the **staggered 1D** instance (data + CP warm start + owner map) for `dd_solve_1d` |
| `dump_data_2d.py` | write the **staggered 2D** instance for `dd_solve_2d` |
| `plot_slurm.py`   | publication figures from a SLURM result directory of `dd_solve_2d --save-solution` files (self-contained: only numpy + matplotlib) |

`mpcc_utils.py` is a self-contained extraction; the full reference solvers
(IPOPT continuation driver, certificate, and the multi-panel probe/arrowhead
plotters) are **not** part of this archival package — they live in the project's
development repository.

## Setup

Managed with [uv](https://docs.astral.sh/uv/). From this directory:

```bash
uv sync          # creates .venv, installs the locked versions
```

That is the whole setup — `uv` reads `.python-version` and fetches CPython 3.11
itself if it is not already present, so no system Python is required. Then run
anything with `uv run`, which re-syncs first if the lock has moved:

```bash
uv run python dump_data_1d.py --n 64 --nsub 4 -o ../cpp/data/data_1d_64.txt
```

(`. .venv/bin/activate` also works if you prefer a plain `python` on the path.)

| file | role |
|---|---|
| `pyproject.toml`   | source of truth for dependencies and the Python floor |
| `.python-version`  | pins CPython 3.11 — what the reference formulation was validated on |
| `uv.lock`          | exact resolved versions, committed so a run is reproducible |
| `requirements.txt` | **generated** pip fallback for environments without uv |

Dependencies are pure numpy/scipy/PIL/matplotlib — no IPOPT or cyipopt, since
the solving happens in `../cpp`.

<details>
<summary>Without uv</summary>

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is exported from `uv.lock`, so the versions match. Regenerate
it after any dependency change:

```bash
uv export --format requirements-txt --no-hashes --no-emit-project -o requirements.txt
```

</details>

> The Python version is pinned deliberately. These scripts exist to produce
> **byte-identical** solver inputs, and that guarantee is only as good as the
> environment that generated the recorded instances — see the check in
> [`../docs/RELEASE.md`](../docs/RELEASE.md).

## Generate an instance, solve it in C++, plot the result

The dump scripts are the **only** route where Python and C++ solve a
byte-identical problem (NumPy's RNG has no C++ equivalent, so an independently
generated instance would differ). Example, 1D:

```bash
# 1) build the instance (data + Chambolle–Pock warm start + DD owner map)
uv run python dump_data_1d.py --n 64 --nsub 4 -o ../cpp/data/data_1d_64.txt

# 2) solve it in C++ (see ../cpp/README.md)
(cd ../cpp && ./dd_solve_1d --data data/data_1d_64.txt --nsub 4 --solver dd)
```

2D, from the bundled image or a phantom:

```bash
uv run python dump_data_2d.py --N 16 --nsub 2 -o ../cpp/data/data_2d_16.txt           # cameraman
uv run python dump_data_2d.py --phantom --N 16 --nsub 2 -o ../cpp/data/phantom_16.txt # synthetic
```

(The uniform 2D driver `dd_solve` and both staggered drivers can also take a PNG
directly — `--data image.png --size N` — using the C++ side's own warm start; the
dump route is for exact Python↔C++ comparison.)

## Plotting

`plot_slurm.py` renders the 2D result figures (noisy / reconstruction / dual
radius δ / MPCC index sets / complementarity residual / continuation path) from
the self-contained `dd_solve_2d --save-solution` files a batch run produces:

```bash
uv run python plot_slurm.py ../cpp/results/slurm_<id>   # one PNG set per solution in the dir
```

## The bundled test image

The dump scripts default `--data` to `../images/cameraman.png` (package root).
Pass `--data <path>` for another image.
