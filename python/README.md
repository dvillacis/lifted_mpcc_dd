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
| `plot_dataset.py` | figures and a booktabs table for **dataset** runs (`dd_solve_dataset`): per-pair gain, contact sheet, and cross-run scaling. Reuses `plot_slurm.py`'s decoder by import |
| `plot_style.py`   | shared figure style: `screen` (200-dpi PNG) and `paper` (two-column PDF) presets, Okabe–Ito palette, figure sizing helpers |
| `aggregate_runs.py` | read the `timings.csv` a sweep writes into one tidy table — works for the **1D and 2D sweeps too**, not just dataset runs |

`mpcc_utils.py` is a self-contained extraction; the full reference solvers
(IPOPT continuation driver, certificate, and the multi-panel probe/arrowhead
plotters) are **not** part of this archival package — they live in the project's
development repository.

## Plotting a local `dd_solve_2d` run

`plot_slurm.py` takes a directory containing a `sols/` subfolder, so a local
run only needs that layout — no SLURM involved:

```bash
mkdir -p runs/sols
./dd_solve_2d --data ../images/cameraman.png --size 32 --nsub 3 \
    --solver ddsimple --hessian exact --save-solution runs/sols/sol_N32_k3.txt
uv run python plot_slurm.py runs          # -> runs/plots/N32_k3_*.png
```

Use a `.npz` extension (the structured format: named, shaped arrays — see
`cpp/README.md`); `.txt` still works for older files, and a directory may
hold both (same tag, `.npz` wins). Decoding either gives the same dict, so
`parse_solution` is the one entry point for both.

Seven panels per solution (noisy, recon, diff, delta, indexsets, residual,
continuation); `--only TAG` restricts to one, `--diff residual` switches the
difference panel to `u−f`, `--out DIR` redirects. The solution file is
self-contained — it carries the instance and the μ-trace — so nothing else is
needed. Both `--formulation` variants write the same format.

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
uv run python plot_slurm.py ../results/slurm_<id>      # one PNG set per solution in the dir
```

### Dataset runs

A dataset run learns ONE α across S training pairs, so its interesting
quantities are **across** pairs. `plot_slurm.py` still answers "what did this
image look like", but as the primary report it is the wrong unit of analysis:
7×S files, of which S are byte-identical continuation plots (the level history
and μ-trace describe the one shared solve), each panel scaled per-file so a
montage invites the wrong comparison. `plot_dataset.py` works from the run's
`--save-report` JSON instead:

```bash
# one run: per-pair gain, contact sheet, and ONE continuation figure
uv run python plot_dataset.py --run ../results/slurm_<id>/reports/report_<tag>.json \
    --sols ../results/slurm_<id>/sols --format both

# many runs: scaling, DD-vs-monolithic, and tab_dataset.tex
uv run python plot_dataset.py --sweep ../results/slurm_<id> --format both

# the timings.csv reader — also works on existing 1D/2D sweeps
uv run python aggregate_runs.py ../results/slurm_<id> -o runs.csv
uv run python aggregate_runs.py ../results            # every slurm_* under it
```

Three things worth knowing:

- `<tag>_gain` is the headline. A dumbbell per pair from noisy PSNR to the PSNR
  at the learned α, sorted by gain, training filled and held-out hollow. The
  mean is not the story — the SPREAD is (measured on Kodak N=32: +8.21 dB on
  one image, +2.16 dB on another, from the same α).
- `<tag>_contact` computes its gray range and its error limit over **all**
  displayed pairs, unlike `plot_slurm.py`'s per-file scaling, and selects from
  both roles proportionally so `--max-pairs` never drops the entire held-out
  half. Anything omitted is listed, never dropped silently.
- `dd_vs_mono`'s right panel is a correctness check drawn as a figure: every
  solver must land on the same α*, so visible separation is a bug. It prints the
  max relative spread (measured 1.0e-09 across ma57 / DD / DD+tiles at four S).

`plot_slurm.py` is deliberately **not** ported onto `plot_style.py` — its
figures are recorded output, and restyling it would silently change every
figure already in `results/`.

## The bundled test images

| image | size | use |
|---|---|---|
| `images/cameraman.png` | 512×512 | the default; native at `--size 512` |
| `images/mariposa.png`  | 1184×1184 | large-N runs — area-downsamples cleanly to 1024 |

The dump scripts default `--data` to `../images/cameraman.png` (package root).
Pass `--data <path>` for another image. The 2D batch scripts prefer
`mariposa.png` when present, since it is the only bundled source big enough for
`--size 1024` without upsampling.
