import subprocess
import re
import os

# Instancias a ejecutar
instancias = [
    ("cameraman, scalar", "cameraman.png", 4, 0),
    ("cameraman, $4\\times4$", "cameraman.png", 4, 4),
    ("cameraman, $8\\times8$", "cameraman.png", 4, 8),
    ("cameraman, $16\\times16$", "cameraman.png", 4, 16),
    ("mariposa, $8\\times8$", "mariposa.png", 4, 8),
    ("phantom, $8\\times8$", "phantom64.png", 4, 8),
    ("16 BSDS crops, $8\\times8$", "bsds64", 1, 8)  
]

def ejecutar_run_learn(args):
    """Ejecuta run_learn.sh inyectando TMPDIR."""
    env = os.environ.copy()
    env["TMPDIR"] = "/tmp"  
    
    cmd = ["./run_learn.sh"] + [str(x) for x in args]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        return res.stdout + "\n" + res.stderr
    except Exception as e:
        print(f"Error crítico ejecutando: {' '.join(cmd)}\n{e}")
        return ""

def buscar(regex, texto, default="NaN"):
    match = re.search(regex, texto)
    return match.group(1) if match else default

def formatear_loss(valor):
    """Aplica el formato de 6 decimales solo si el valor es un número válido."""
    try:
        return f"{float(valor):.6f}"
    except ValueError:
        return valor

def main():
    print("Executing denoising experiments ...\n")
    print("-" * 50)
    
    latex_output = [
        r"\begin{table}[ht]",
        r"  \centering\small",
        r"  \begin{tabular}{@{}lrrrrr@{}}",
        r"    \toprule",
        r"    & MPCC alone & stage 1 alone & \multicolumn{3}{c}{two stages}\\ ",
        r"    \cmidrule(l){4-6}",
        r"    instance & loss (its, s) & loss (s) & loss & its & stage 1 + 2 (s)\\ ",
        r"    \midrule"
    ]

    for etiqueta, img, nsub, block in instancias:
        print(f"Processing: {etiqueta}...")

        # Configuración estándar para todas las instancias (denoising)
        base_sh = [
            "-i", img, "-n", 64, "-k", nsub, "-p", 4, 
            "--block", block, "--sigma", 0.1, "--force"
        ]

        # -------------------------------------------------------------
        # 1. MPCC alone
        # -------------------------------------------------------------
        args_uni = base_sh + ["--start", "uniform", "--t-min", "1e-4", "--", "--gate", "kkt"]
        log_uni = ejecutar_run_learn(args_uni)
        
        uni_loss = formatear_loss(buscar(r'upper-level loss\s+([\d.]+)', log_uni))
        uni_its  = buscar(r'after (\d+) iterations', log_uni, "NaN")
        uni_time = buscar(r'time\s+([\d.]+)\s*s', log_uni, "NaN")

        # -------------------------------------------------------------
        # 2. Stage 1 alone
        # -------------------------------------------------------------
        args_stg1 = base_sh + [
            "--start", "smoothed", "--max-iter", 0, 
            "--", "--start-gamma", "1e-3", "--start-iters", 100
        ]
        log_stg1 = ejecutar_run_learn(args_stg1)
        
        stg1_loss = formatear_loss(buscar(r'upper-level loss\s+([\d.]+)', log_stg1))
        stg1_time = buscar(r'Stage 1.*?[+|=]?\s*([\d.]+)\s*s', log_stg1, "NaN")

        # -------------------------------------------------------------
        # 3. Two stages
        # -------------------------------------------------------------
        args_two = base_sh + [
            "--start", "smoothed", "--t-min", "1e-4", 
            "--", "--gate", "kkt", "--start-gamma", "1e-3", "--start-iters", 100
        ]
        log_two = ejecutar_run_learn(args_two)
        
        two_loss  = formatear_loss(buscar(r'upper-level loss\s+([\d.]+)', log_two))
        two_its   = buscar(r'after (\d+) iterations', log_two, "NaN")
        
        t1 = buscar(r'stage 1\s+([\d.]+)\s*s', log_two, "NaN")
        t2 = buscar(r'stage 2\s+([\d.]+)\s*s', log_two, "NaN")

        if stg1_time == "NaN" and t1 != "NaN":
            stg1_time = t1

        fila = (f"    {etiqueta} "
                f"& {uni_loss} ({uni_its}, {uni_time}) "
                f"& {stg1_loss} ({stg1_time}) "
                f"& {two_loss} & {two_its} & {t1} + {t2}\\\\")
        
        latex_output.append(fila)

    latex_output.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Draft (Auto-generado). Exact upper-level loss, $N=64$, $\sigma=0.1$, $4\times4$ tiles per image, 4 ranks, \texttt{--gate kkt}, $t_{\min}=10^{-4}$. Stage 1: Huber $\gamma=10^{-3}$, at most 100 iterations.}\label{tab:stages}",
        r"\end{table}"
    ])

    print("\n" + "=" * 50)
    print("CÓDIGO LATEX GENERADO:")
    print("=" * 50 + "\n")
    print("\n".join(latex_output))

if __name__ == "__main__":
    main()