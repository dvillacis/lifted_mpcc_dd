#!/usr/bin/env python3
"""Create a compact LaTeX results table from one or more tv_dd run logs."""
import argparse
import re
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent


def resolve_input(path):
    path = Path(path).expanduser()
    if path.is_absolute() or path.exists():
        return path
    repo_relative = REPO_DIR / path
    if repo_relative.exists():
        return repo_relative
    return path


def resolve_output(path):
    path = Path(path).expanduser()
    if path.is_absolute() or path.parent.exists():
        return path
    return REPO_DIR / path


def required(pattern, text, label):
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise ValueError(f"could not find {label}")
    return match


def parse_log(path):
    path = resolve_input(path)
    text = path.read_text(errors="replace")
    csv_line = required(r"^CSV,(.*)$", text, "CSV summary line").group(1)
    fields = csv_line.split(",")
    if len(fields) < 26:
        raise ValueError(f"incomplete CSV line in {path}")

    image, n, nsub = fields[0], int(fields[1]), int(fields[2])
    tmin = float(fields[5])
    ranks = int(fields[8])
    ipm_iters = int(fields[12])
    psnr = float(fields[17])
    peak_mb = int(float(fields[25]))
    factor_count = int(fields[21])

    s1 = required(
        r"Stage 1 (?:converged|stopped).*?: (\d+) iterations, (\d+) lower-level solves, ([\d.]+) s",
        text,
        "stage 1 summary",
    )
    stage1_iters, lower_solves, stage1_s = s1.groups()
    s2 = required(r"Stage 2 level gate .*? after (\d+) iterations, t = ([\d.eE+-]+), ([\d.]+) s", text,
                  "stage 2 level-gate summary")
    stage2_iters, _, stage2_s = s2.groups()
    total = required(r"time\s+stage 1 [\d.]+ s \+ stage 2 [\d.]+ s = ([\d.]+) s", text,
                     "total time").group(1)
    kkt = required(r"KKT dimension ([\d,]+)", text, "KKT dimension").group(1).replace(",", "")
    stop = "level gate" if "[level gate]" in text else "other"
    tmin_tex = f"{tmin:.0e}"
    mantissa, exponent = tmin_tex.split("e")
    exponent = int(exponent)
    tmin_tex = rf"{mantissa}\times10^{{{exponent}}}"
    return {
        "N": n,
        "tiles": f"{nsub}\\times{nsub}",
        "ranks": ranks,
        "tmin": tmin_tex,
        "kkt": f"{int(kkt):,}",
        "stage1": f"{stage1_iters} / {lower_solves}; {stage1_s}",
        "stage2": f"{stage2_iters} / {factor_count}; {stage2_s}",
        "total": total,
        "memory": peak_mb,
        "psnr": f"{psnr:.2f}",
        "stop": stop,
        "image": image,
    }


def latex_escape(value):
    return str(value).replace("_", r"\_")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="tv_dd log files")
    parser.add_argument("-o", "--output", required=True, help="output .tex file")
    args = parser.parse_args()
    rows = [parse_log(p) for p in args.logs]
    rows.sort(key=lambda row: row["N"])

    run_settings = "; ".join(
        f"$N={row['N']}$: $t_{{\\min}}={row['tmin']}$, {row['stop']} stop"
        for row in rows
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        (r"\caption{Two-stage results for Mariposa (" + run_settings +
         r"). Stage 1 runs on rank 0; stage columns report iterations / seconds (stage 2 iterations are IPM iterations). "
         r"Memory is peak per rank over the full run.}"),
        r"\label{tab:mariposa-scaling}",
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"$N$ & Tiles & Ranks & KKT dim. & \shortstack{Stage 1\\(it./s)} & \shortstack{Stage 2\\(it./s)} & \shortstack{Memory\\(MB/rank)} & \shortstack{PSNR\\(dB)} \\",
        r"\midrule",
    ]
    for row in rows:
        s1 = row["stage1"].split(";")
        s2 = row["stage2"].split(";")
        values = [
            row["N"],
            f"${row['tiles']}$",
            row["ranks"],
            f"{int(row['kkt'].replace(',', '')):,}".replace(",", r"{,}"),
            f"{s1[0].split('/')[0].strip()} / {s1[1].strip()}",
            f"{s2[0].split('/')[0].strip()} / {s2[1].strip()}",
            row["memory"],
            row["psnr"],
        ]
        lines.append(" & ".join(latex_escape(x) for x in values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    output = resolve_output(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"LaTeX table: {output}")


if __name__ == "__main__":
    main()
