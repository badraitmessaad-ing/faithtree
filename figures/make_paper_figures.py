"""Recreate every figure of the FaithTree paper from the result files in this repository.

    python figures/make_paper_figures.py            # Figs. 2-7  -> figures/out/
    python figures/make_paper_figures.py --fig1     # also renders Fig. 1 (needs Node.js and LibreOffice)

Inputs (all in the repository, no dataset needed):
    esp32/serial_logs/*.txt                         board runs (Figs. 3 and 4)
    figures/data/confusion_matrix_saved_model.csv   test confusion matrix of the saved NF-ToN-IoT-v2 tree (Fig. 5)
    figures/data/tab07_optimizer_comparison.csv     tuning comparison written by notebook 04 (Fig. 6)
    figures/data/generalization_per_class.csv       random vs unseen-host recall written by notebook 12 (Fig. 7)
Fig. 2 is a schematic and has no data. Every number drawn in Figs. 3-7 is read from these files.
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "esp32" / "serial_logs"
DATA = ROOT / "figures" / "data"

BLUE, ORANGE, GRAY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#8a8986", "#0b0b0b", "#52514e", "#e6e5e2"
STYLE = {"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 300}


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------- serial logs
def read_log(path):
    """Return (summary dict, per-flow DataFrame) from a benchmark serial log."""
    summary, rows, header = {}, [], None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("sample_id,"):
            header = line.split(",")
        elif line[:1].isdigit() and header:
            rows.append(line.split(","))
        elif "," in line and not line.startswith("#"):
            k, v = line.split(",", 1)
            summary[k] = v
    df = pd.DataFrame(rows, columns=header).apply(pd.to_numeric)
    return summary, df


# ----------------------------------------------------------------------------- Fig. 2
def fig2_threshold(out):
    """Schematic: nearest-float32 rounding (legacy) versus the exact raw-space threshold."""
    fig, ax = plt.subplots(figsize=(3.5, 1.25))
    ax.axis("off")
    ax.hlines(0, -0.35, 1.35, color=INK2, lw=1)
    for x, label in [(0, "a (float32)"), (1, "b (next float32)")]:
        ax.plot(x, 0, "|", ms=14, mew=1.5, color=INK)
        ax.text(x, -0.32, label, ha="center", va="top", fontsize=7, color=INK)
    t64 = 0.64
    ax.plot(t64, 0.0, "v", ms=7, color=INK, zorder=3)
    ax.text(t64, 0.36, "scikit-learn threshold t (float64)", ha="center", fontsize=7, color=INK)
    ax.annotate("", xy=(0.985, 0.12), xytext=(t64, 0.12), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))
    ax.annotate("", xy=(0.015, -0.12), xytext=(t64, -0.12), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    ax.text(1.12, 0.16, "legacy: nearest float32", fontsize=6.6, color=ORANGE, va="center")
    ax.text(1.12, -0.14, "exact: largest float32 ≤ t", fontsize=6.6, color=BLUE, va="center")
    ax.set_xlim(-0.4, 2.0)
    ax.set_ylim(-0.6, 0.5)
    fig.tight_layout()
    save(fig, out, "fig_exact_threshold_col")


# ----------------------------------------------------------------------------- Fig. 3
def fig3_latency(out):
    """Cumulative distribution of the single-call inference time of the 5,000 board flows."""
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    for log, color, name in [("nf_ton_iot_v2_serial.txt", BLUE, "NF-ToN-IoT-v2"),
                             ("nf_unsw_nb15_v3_serial.txt", ORANGE, "NF-UNSW-NB15-v3")]:
        _, df = read_log(LOGS / log)
        v = df["infer_us"].to_numpy()
        n_nodes = next((l.split(",")[2] for l in (LOGS / log).read_text().splitlines() if l.startswith("#META,n_nodes")), "?")
        s = np.sort(v)
        ax.step(s, np.arange(1, len(s) + 1) / len(s), where="post", color=color, lw=1.6,
                label=f"{name} ({int(n_nodes):,} nodes)")
        ax.plot(np.percentile(v, 99), 0.99, "o", ms=4.5, color=color, mec="white", mew=0.8, zorder=3)
    ax.set_xlabel("inference time per flow (µs)")
    ax.set_ylabel(f"fraction of {len(df):,} flows")
    ax.set_xlim(0, 48)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    save(fig, out, "fig_latency")


# ----------------------------------------------------------------------------- Fig. 4
def fig4_exporters(out):
    """Exporters on the ESP32: batch mean (bar), p99 (tick) and agreement, read from the three toolchain logs."""
    rows = []
    for label, log, color in [("Exact export", "toolchain_prod_ours.txt", BLUE),
                              ("Legacy export", "toolchain_prod_ours_LEGACY.txt", GRAY),
                              ("emlearn 0.23.2", "toolchain_prod_emlearn.txt", GRAY)]:
        summary, _ = read_log(LOGS / log)
        rows.append((label, float(summary["INFER_US_MEAN_BATCH"]), float(summary["INFER_US_P99"]),
                     summary["SUMMARY_MATCHES"], color))
    fig, ax = plt.subplots(figsize=(3.5, 1.7))
    for i, (_, mean, p99, agree, color) in enumerate(rows[::-1]):
        ax.barh(i, mean, height=0.5, color=color, zorder=2)
        ax.plot([mean, p99], [i, i], color=INK2, lw=1, zorder=3)
        ax.plot(p99, i, "|", color=INK2, ms=7, mew=1.1, zorder=3)
        n_ok, n_all = agree.split("/")
        ax.text(p99 + 1.2, i, f"{int(n_ok):,}/{int(n_all):,}", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]], color=INK, fontsize=7.5)
    ax.set_xlabel("time per flow (µs): bar = mean, tick = p99", fontsize=7.5)
    ax.set_xlim(0, 68)
    ax.grid(axis="y", visible=False)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, out, "fig_exporter_board_col")


# ----------------------------------------------------------------------------- Fig. 5
def fig5_confusion(out):
    """Row-normalized test confusion matrix of the saved NF-ToN-IoT-v2 tree."""
    cm = pd.read_csv(DATA / "confusion_matrix_saved_model.csv", index_col=0)
    labels = [c.replace("true_", "") for c in cm.index]
    m = cm.values.astype(float)
    m = m / m.sum(1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("b", ["#ffffff", "#a9c9ef", BLUE, "#0d3f7a"])
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.grid(False)
    ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(m)):
        for j in range(len(m)):
            if m[i, j] >= 0.01:
                txt = f"{m[i, j]:.2f}".lstrip("0") if m[i, j] < 1 else "1.0"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6, color="white" if m[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.spines[:].set_visible(False)
    save(fig, out, "fig_confusion")


# ----------------------------------------------------------------------------- Fig. 6
def fig6_tuning(out):
    """Validation MCC of the tuning methods: median (dot) and IQR (line); filled = at least 80% of runs deployable."""
    t = pd.read_csv(DATA / "tab07_optimizer_comparison.csv")
    t.columns = ["Dataset", "Method", "Runs", "med", "q25", "q75", "nodes", "kb", "rank", "dep"]
    rename = {"Full (S→D→P)": "Silverback→Dolphin→Prune", "Full + restricted space": "Restricted space",
              "Full + restricted space + MCC objective": "Restricted + MCC objective",
              "Full + restricted + MCC + no novelty bonus": "Restricted + MCC, no novelty bonus",
              "Halving-Silverback (aligned)": "Halving variant (restricted)",
              "Random search (aligned)": "Random search (restricted)", "Pruning-grid (aligned)": "Grid (restricted)",
              "Pruning-grid search": "Pruning grid (full space)", "Random search": "Random search (full space)",
              "Default DT (no search)": "Default tree (no search)", "Optuna TPE": "Optuna TPE"}
    t["Label"] = t.Method.map(lambda m: rename.get(m, m))
    groups = [("Original metaheuristic", ORANGE, ["Silverback only", "Silverback→Dolphin", "Silverback→Prune",
                                                  "Silverback→Dolphin→Prune"]),
              ("Metaheuristic, restricted space", BLUE, ["Restricted space", "Restricted + MCC objective",
                                                          "Restricted + MCC, no novelty bonus", "Halving variant (restricted)"]),
              ("Simple baselines", GRAY, ["Random search (full space)", "Optuna TPE", "Pruning grid (full space)",
                                          "Random search (restricted)", "Grid (restricted)", "Default tree (no search)"])]
    order, ypos, heads, y = [], {}, [], 0
    for g, c, methods in groups:
        heads.append((y, g, c))
        y += 1
        for m in methods:
            ypos[m] = y
            order.append((m, c))
            y += 1
        y += 0.4
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.1), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (ds, title) in zip(axes, [("NF-ToN-IoT-v2", "NF-ToN-IoT-v2 (paper scale, 5 repeats)"),
                                      ("NF-UNSW-NB15-v3", "NF-UNSW-NB15-v3 (reduced scale, 10 repeats)")]):
        d = t[t.Dataset == ds].set_index("Label")
        for m, c in order:
            if m not in d.index:
                continue
            r = d.loc[m]
            yy = ypos[m]
            ax.plot([r.q25, r.q75], [yy, yy], color=c, lw=2.2, solid_capstyle="round", zorder=2)
            ax.plot(r.med, yy, "o", ms=6.5, mfc=c if r.dep >= 0.8 else "white", mec=c, mew=1.5, zorder=3)
        ax.axvline(d.med.max(), color=INK2, lw=0.7, ls=(0, (3, 3)), zorder=1)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("validation MCC (median, IQR)")
        ax.grid(axis="y", visible=False)
        for hy, _, _ in heads:
            ax.axhspan(hy - 0.5, hy + 0.5, color="#f3f2ef", zorder=0, lw=0)
    axes[0].set_yticks([ypos[m] for m, _ in order] + [h[0] for h in heads])
    axes[0].set_yticklabels([m for m, _ in order] + [h[1] for h in heads], fontsize=7.8)
    for lab in axes[0].get_yticklabels():
        for _, g, c in heads:
            if lab.get_text() == g:
                lab.set_fontweight("bold")
                lab.set_color(c)
    axes[0].invert_yaxis()
    for a in axes:
        a.tick_params(axis="y", length=0)
    handles = [Line2D([], [], color=INK2, marker="o", lw=0, ms=6.5, label="≥ 80% of runs deployable"),
               Line2D([], [], color=INK2, marker="o", mfc="white", lw=0, ms=6.5, label="< 80% of runs deployable"),
               Line2D([], [], color=INK2, lw=0.7, ls=(0, (3, 3)), label="best median")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=3, frameon=False, fontsize=7.8)
    fig.subplots_adjust(left=0.27, right=0.99, top=0.94, bottom=0.13, wspace=0.08)
    save(fig, out, "fig_tuning")


# ----------------------------------------------------------------------------- Fig. 7
def fig7_unseen_host(out):
    """Per-class recall: random split versus unseen-host split (mean over repeats; line = range)."""
    pc = pd.read_csv(DATA / "generalization_per_class.csv")
    pc = pc[pc.features == "all features"].groupby(["protocol", "class"]).recall.agg(["mean", "min", "max"]).reset_index()
    r = pc[pc.protocol.str.startswith("random")].set_index("class")
    u = pc[pc.protocol.str.startswith("unseen")].set_index("class")
    order = u["mean"].sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for i, c in enumerate(order):
        ax.plot([u.loc[c, "min"], u.loc[c, "max"]], [i, i], color=ORANGE, lw=1.4, alpha=0.45, solid_capstyle="round")
        ax.plot([r.loc[c, "mean"], u.loc[c, "mean"]], [i, i], color=GRID, lw=1, zorder=1)
        ax.plot(r.loc[c, "mean"], i, "o", ms=5, color=GRAY, mec="white", mew=0.8, zorder=3)
        ax.plot(u.loc[c, "mean"], i, "o", ms=5, color=ORANGE, mec="white", mew=0.8, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("recall (mean over 5 repeats)")
    ax.plot([], [], "o", color=GRAY, label="random split")
    ax.plot([], [], "o", color=ORANGE, label="unseen-host split (line = range)")
    ax.legend(frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.45, 1.13), ncol=2,
              handletextpad=0.2, columnspacing=0.8)
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig_unseen_host")


# ----------------------------------------------------------------------------- Fig. 1
def fig1_pipeline(out, soffice):
    """Build the pipeline slide with figures/fig1/build_fig1.js, convert it to PDF with LibreOffice,
    rasterize it at 300 dpi and crop the white margin."""
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz  # older PyMuPDF
    from PIL import Image, ImageChops
    here = ROOT / "figures" / "fig1"
    pptx = out / "fig1_pipeline.pptx"
    if not (here / "node_modules" / "pptxgenjs").exists():
        print("installing the Node packages for Fig. 1 (npm install, once)")
        subprocess.run("npm install", check=True, cwd=here, shell=True)
    subprocess.run(["node", str(here / "build_fig1.js"), str(pptx)], check=True, cwd=here)
    subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(out), str(pptx)], check=True)
    page = fitz.open(out / "fig1_pipeline.pdf")[0]
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    box = ImageChops.difference(img, Image.new("RGB", img.size, "white")).getbbox()
    pad = 20
    img = img.crop((max(box[0] - pad, 0), max(box[1] - pad, 0), min(box[2] + pad, img.width), min(box[3] + pad, img.height)))
    img.save(out / "fig_framework_icons.png", dpi=(300, 300))
    print("wrote fig_framework_icons")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "figures" / "out"))
    ap.add_argument("--fig1", action="store_true", help="also build Fig. 1 (Node.js, LibreOffice and PyMuPDF required)")
    ap.add_argument("--soffice", default=shutil.which("soffice") or r"C:\Program Files\LibreOffice\program\soffice.exe")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    fig2_threshold(out)
    plt.rcParams.update({"font.size": 8})
    fig3_latency(out)
    plt.rcParams.update({"font.size": 8.5})
    fig4_exporters(out)
    plt.rcParams.update({"font.size": 8})
    fig5_confusion(out)
    plt.rcParams.update({"font.size": 8.5})
    fig6_tuning(out)
    plt.rcParams.update({"font.size": 8})
    fig7_unseen_host(out)
    if a.fig1:
        fig1_pipeline(out, a.soffice)
