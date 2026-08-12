"""Aggregate the ``timings.csv`` files a sweep writes into one tidy table.

Every SLURM script here records ``tag,solver,wall_s,cpu_s,maxrss_MB,exit`` per
run through its ``run_timed`` helper, and until now nothing read those files
back: they were pretty-printed into ``summary.txt`` with ``column -t`` and that
was the end of it. So the wall/CPU/peak-RSS instrumentation has never actually
produced an analysis. This script is the missing reader, and it deliberately
covers the EXISTING 1D and 2D sweeps as well as dataset runs — the schema is the
same for all of them.

The tag is the only place a run records its configuration, so it is parsed into
columns. Three forms are recognised, in order:

    <img>_N<n>_S<s>_<label>     dataset sweep  (run_dataset.slurm)
    <img>_N<n>_k<k>             2D sweep       (run_2d*.slurm)
    n<n>_k<k>                   1D sweep       (run_1d.slurm)

Anything else keeps its tag and leaves the parsed columns empty rather than
guessing. Where a dataset run also wrote ``reports/*.json``, its solver-side
numbers (iterations, alpha*, PSNR) are joined in on ``(tag, solver)``.

    uv run python aggregate_runs.py ../results/slurm_4266
    uv run python aggregate_runs.py ../results            # every slurm_* under it
    uv run python aggregate_runs.py ../results -o runs.csv --sort wall_s
"""

import argparse
import csv
import glob
import json
import os
import re

COLUMNS = ["run", "tag", "solver", "image", "N", "S", "k", "wall_s", "cpu_s",
           "maxrss_MB", "exit", "iters", "alpha", "Q_alpha", "psnr_train",
           "psnr_val", "p_border", "n_sub", "kkt_dim"]

_RE_DATASET = re.compile(r"^(?P<image>.+)_N(?P<N>\d+)_S(?P<S>\d+)(?:_(?P<lbl>.+))?$")
_RE_2D = re.compile(r"^(?P<image>.+)_N(?P<N>\d+)_k(?P<k>\d+)$")
_RE_1D = re.compile(r"^n(?P<N>\d+)_k(?P<k>\d+)$")


def parse_tag(tag):
    """tag -> {image, N, S, k}; unknown shapes give empty fields, never guesses."""
    out = {"image": "", "N": "", "S": "", "k": ""}
    m = _RE_DATASET.match(tag)
    if m:
        out.update(image=m.group("image"), N=m.group("N"), S=m.group("S"))
        return out
    m = _RE_2D.match(tag)
    if m:
        out.update(image=m.group("image"), N=m.group("N"), k=m.group("k"))
        return out
    m = _RE_1D.match(tag)
    if m:
        out.update(N=m.group("N"), k=m.group("k"))
        return out
    return out


def _num(s):
    s = (s or "").strip()
    try:
        return float(s)
    except ValueError:
        return None            # the scripts write NA when GNU time is absent


def read_timings(run_dir):
    path = os.path.join(run_dir, "timings.csv")
    if not os.path.isfile(path):
        return []
    run = os.path.basename(os.path.normpath(run_dir))
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if not r.get("tag"):
                continue
            row = dict.fromkeys(COLUMNS, "")
            row.update(run=run, tag=r["tag"], solver=r.get("solver", ""),
                       wall_s=_num(r.get("wall_s")), cpu_s=_num(r.get("cpu_s")),
                       maxrss_MB=_num(r.get("maxrss_MB")), exit=_num(r.get("exit")))
            row.update(parse_tag(r["tag"]))
            rows.append(row)
    return rows


def join_reports(run_dir, rows):
    """Fold dataset --save-report JSON into the rows that match (tag, solver)."""
    reps = {}
    for p in glob.glob(os.path.join(run_dir, "reports", "*.json")):
        try:
            with open(p) as fh:
                d = json.load(fh)
        except (OSError, ValueError) as e:
            print(f"  warning: skipping {p}: {e}")
            continue
        reps[d.get("tag", "")] = d
    if not reps:
        return rows
    for row in rows:
        d = reps.get(row["tag"])
        if not d:
            continue
        tr = [p["psnr_rof"] for p in d.get("pairs", []) if p["role"] == "train"]
        va = [p["psnr_rof"] for p in d.get("pairs", []) if p["role"] == "val"]
        row.update(
            iters=d.get("iters", ""), alpha=d.get("alpha", ""),
            Q_alpha=d.get("Q_alpha", ""), p_border=d.get("p_border", ""),
            n_sub=d.get("n_sub", ""), kkt_dim=d.get("kkt_dim", ""),
            S=row["S"] or d.get("S", ""), N=row["N"] or d.get("N", ""),
            k=row["k"] or d.get("nsub", ""),
            psnr_train=round(sum(tr) / len(tr), 4) if tr else "",
            psnr_val=round(sum(va) / len(va), 4) if va else "")
    return rows


def collect(root):
    """A single run dir, or a parent holding many slurm_* dirs."""
    dirs = []
    if os.path.isfile(os.path.join(root, "timings.csv")):
        dirs = [root]
    else:
        dirs = sorted(d for d in glob.glob(os.path.join(root, "*"))
                      if os.path.isfile(os.path.join(d, "timings.csv")))
    if not dirs:
        raise SystemExit(f"no timings.csv in {root!r} or its immediate children")
    rows = []
    for d in dirs:
        r = join_reports(d, read_timings(d))
        print(f"  {os.path.basename(os.path.normpath(d))}: {len(r)} run(s)")
        rows.extend(r)
    return rows


def print_table(rows, cols=None):
    cols = cols or [c for c in COLUMNS
                    if any(r.get(c) not in ("", None) for r in rows)]
    def cell(r, c):
        v = r.get(c, "")
        if v is None:
            return "NA"
        if isinstance(v, float):
            # %g at 6 significant digits: keeps alpha readable (0.0655939)
            # instead of rounding it to 0.07, and prints whole numbers whole
            # (exit codes, MB) instead of 0.00 / 34.00.
            return f"{v:.0f}" if v == int(v) else f"{v:.6g}"
        return str(v)
    w = {c: max(len(c), max((len(cell(r, c)) for r in rows), default=0))
         for c in cols}
    print("  ".join(c.rjust(w[c]) for c in cols))
    print("  ".join("-" * w[c] for c in cols))
    for r in rows:
        print("  ".join(cell(r, c).rjust(w[c]) for c in cols))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="a results dir with timings.csv, or a parent of "
                                 "several slurm_* dirs")
    ap.add_argument("-o", "--out", default=None,
                    help="write the tidy CSV here (default: print only)")
    ap.add_argument("--sort", default=None,
                    help="column to sort by (e.g. wall_s, S, tag)")
    args = ap.parse_args()

    rows = collect(args.root)
    if args.sort:
        if args.sort not in COLUMNS:
            raise SystemExit(f"--sort must be one of: {', '.join(COLUMNS)}")
        rows.sort(key=lambda r: (r.get(args.sort) is None,
                                 _sortable(r.get(args.sort))))
    print()
    print_table(rows)
    if args.out:
        d = os.path.dirname(args.out)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({c: ("" if r.get(c) is None else r.get(c, ""))
                            for c in COLUMNS})
        print(f"\nsaved → {args.out}   ({len(rows)} rows)")


def _sortable(v):
    if v in ("", None):
        return (1, 0.0, "")
    try:
        return (0, float(v), "")
    except (TypeError, ValueError):
        return (0, 0.0, str(v))


if __name__ == "__main__":
    main()
