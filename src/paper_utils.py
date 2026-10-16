"""
paper_utils.py -- helpers for notebooks 05 (hardware) and 06 (paper figures/tables).
Pure pandas/matplotlib; needs NO raw data and NO trained model.
"""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

DATASETS = ["nf_ton_iot_v2", "cic_ton_iot", "nf_unsw_nb15_v3"]
DS_LABEL = {"nf_ton_iot_v2": "NF-ToN-IoT-v2", "cic_ton_iot": "CIC-ToN-IoT", "nf_unsw_nb15_v3": "NF-UNSW-NB15-v3"}
DS_COLOR = {"nf_ton_iot_v2": "#1f77b4", "cic_ton_iot": "#d95f02", "nf_unsw_nb15_v3": "#2ca02c"}

HIGHLIGHT = "#1f77b4"
NEUTRAL = "#8c8c8c"


def set_paper_style():
    matplotlib.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "figure.constrained_layout.use": True,
    })


def save_fig(fig, path_no_ext):
    """Save PNG (300 dpi) + vector PDF next to each other."""
    p = Path(path_no_ext)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    return p.with_suffix(".png")


# ----------------------------------------------------------------------------- tables
def _tex_escape(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s


def df_to_latex(df, caption="", label="", float_fmt="{:.4f}", index=False):
    """Dependency-free LaTeX (booktabs) writer."""
    d = df.reset_index() if index else df.copy()
    cols = list(d.columns)
    out = ["\\begin{table}[t]", "\\centering", "\\small"]
    if caption:
        out.append(f"\\caption{{{_tex_escape(caption)}}}")
    if label:
        out.append(f"\\label{{{label}}}")
    out += ["\\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}", "\\toprule",
            " & ".join(_tex_escape(c) for c in cols) + " \\\\", "\\midrule"]
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("--" if np.isnan(v) else float_fmt.format(v))
            else:
                cells.append(_tex_escape(v))
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def save_table(df, path_no_ext, caption="", label="", float_fmt="{:.4f}", index=False):
    p = Path(path_no_ext)
    p.parent.mkdir(parents=True, exist_ok=True)
    (df.reset_index() if index else df).to_csv(p.with_suffix(".csv"), index=False)
    p.with_suffix(".tex").write_text(df_to_latex(df, caption, label, float_fmt, index), encoding="utf-8")
    return p


# ----------------------------------------------------------------------------- serial logs
_TS = re.compile(r"^\d{1,2}:\d{2}:\d{2}[.,]\d+\s*->\s*")


def parse_serial_log(path):
    """Parse the Serial-Monitor output of the ESP32 benchmark sketch.

    Understands BOTH
      * the current sketch (#META lines, per-sample infer_us column, INFER_US_* summary)
      * the LEGACY sketch (sample_id,true_label,python_pred,esp32_pred,match + TOTAL_US/US_PER_SAMPLE)
    """
    text = Path(path).read_text(errors="ignore")
    meta, summary, rows, cols = {}, {}, [], None
    for raw in text.splitlines():
        line = _TS.sub("", raw.strip().lstrip("\ufeff"))
        if not line:
            continue
        if line.startswith("#META,"):
            parts = line.split(",", 2)
            if len(parts) == 3:
                meta[parts[1]] = parts[2]
        elif line.startswith("sample_id,"):
            cols = [c.strip() for c in line.split(",")]
        elif cols and re.match(r"^\d+,", line) and line.count(",") == len(cols) - 1:
            rows.append(line.split(","))
        elif re.match(r"^[A-Z][A-Z0-9_]+,", line):
            k, v = line.split(",", 1)
            summary[k] = v.strip()
    df = pd.DataFrame(rows, columns=cols).apply(pd.to_numeric, errors="coerce") if rows else pd.DataFrame()
    return {"meta": meta, "summary": summary, "samples": df,
            "legacy": ("infer_us" not in (cols or [])), "complete": ("#DONE" in text) or ("SUMMARY_MATCHES" in summary)}


def summarize_hardware(dataset, log):
    """One summary row (dict) from a parsed log."""
    df, s, m = log["samples"], log["summary"], log["meta"]
    n = len(df)
    row = {"dataset": dataset, "n_samples": n,
           "python_esp32_matches": int((df["esp32_pred"] == df["python_pred"]).sum()) if n else np.nan,
           "agreement": float((df["esp32_pred"] == df["python_pred"]).mean()) if n else np.nan,
           "esp32_true_label_accuracy_on_subset": float((df["esp32_pred"] == df["true_label"]).mean()) if n else np.nan,
           "n_classes_in_subset": int(df["true_label"].nunique()) if n else np.nan,
           "log_format": "legacy (latency includes Serial.print)" if log["legacy"] else "inference-only timing"}
    if log["legacy"]:
        tot = float(s.get("TOTAL_US", "nan"))
        row.update({"latency_valid": False, "mean_latency_us_serial_inclusive": tot / n if n else np.nan})
    else:
        row.update({"latency_valid": True,
                    "mean_latency_us_batch": float(s.get("INFER_US_MEAN_BATCH", "nan")),
                    "mean_latency_us_single": float(s.get("INFER_US_MEAN_SINGLE", "nan")),
                    "min_us": float(s.get("INFER_US_MIN", "nan")), "median_us": float(s.get("INFER_US_MEDIAN", "nan")),
                    "p95_us": float(s.get("INFER_US_P95", "nan")), "p99_us": float(s.get("INFER_US_P99", "nan")),
                    "max_us": float(s.get("INFER_US_MAX", "nan"))})
        mb = row["mean_latency_us_batch"]
        row["throughput_samples_per_s"] = 1e6 / mb if mb and mb == mb else np.nan
        for k_meta, k_out in [("chip_model", "chip"), ("chip_revision", "chip_rev"), ("cpu_mhz", "cpu_mhz"),
                              ("sketch_size_bytes", "benchmark_firmware_bytes"),
                              ("free_sketch_space_bytes", "app_partition_bytes"),   # ESP.getFreeSketchSpace() = size of the next OTA partition, NOT the free flash
                              ("heap_size_bytes", "heap_size_bytes"), ("free_heap_before", "free_heap_before"),
                              ("free_heap_after", "free_heap_after"), ("min_free_heap", "min_free_heap"),
                              ("n_nodes", "n_nodes"), ("n_selected_features", "n_selected_features")]:
            if k_meta in m:
                row[k_out] = m[k_meta]
    try:
        fw, part = float(row["benchmark_firmware_bytes"]), float(row["app_partition_bytes"])
        row["flash_headroom_bytes"] = part - fw
        row["flash_used_pct"] = 100.0 * fw / part
    except (KeyError, ValueError, TypeError):
        pass
    return row


def parse_footprint(path):
    s = parse_serial_log(path)["summary"]
    return int(s["SKETCH_SIZE_BYTES"]) if "SKETCH_SIZE_BYTES" in s else None


# ----------------------------------------------------------------------------- results loaders
def load_json(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


FLASH_BYTES_PER_NODE = 10   # exact export: int16 left + int16 right + uint8 feature + float32 threshold + uint8 class


def flash_kb(node_count):
    """Model flash of the exact export: 10 B per node (the real layout, not the 24 B proxy used in the search fitness)."""
    return float(node_count) * FLASH_BYTES_PER_NODE / 1024.0


def read_csv_if(p, **kw):
    p = Path(p)
    if not p.exists():
        return None
    df = pd.read_csv(p, **kw)
    # report the real exported size; framework.py keeps its 24 B/node proxy for the search fitness
    if "size_proxy_kb" in df.columns and "node_count" in df.columns:
        df["size_proxy_kb"] = df["node_count"] * FLASH_BYTES_PER_NODE / 1024.0
    return df
