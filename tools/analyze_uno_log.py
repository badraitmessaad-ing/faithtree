"""
analyze_uno_log.py -- summarise the Serial output of uno_dt_test.ino.

    python tools/analyze_uno_log.py --dataset nf_ton_iot_v2_binary [--sketch-bytes 27450]

Reads esp32/serial_logs/<dataset>_uno_serial.txt and writes results/uno_export/<dataset>/uno_summary.json:
agreement with scikit-learn, mean / median / p95 / p99 / max inference time, free RAM, and (if given) the
sketch size reported by the Arduino IDE.
"""
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--sketch-bytes", type=int, default=None, help='"Sketch uses N bytes" from the Arduino IDE')
a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
log = ROOT / "esp32" / "serial_logs" / f"{a.dataset}_uno_serial.txt"
text = log.read_text(errors="ignore")
meta, rows, cols = {}, [], None
for raw in text.splitlines():
    line = re.sub(r"^\d\d:\d\d:\d\d\.\d+ -> ", "", raw.strip())   # Serial Monitor timestamps, if enabled
    if line.startswith("#META,"):
        p = line.split(",", 2)
        if len(p) == 3: meta[p[1]] = p[2]
    elif line.startswith("sample_id,"):
        cols = line.split(",")
    elif cols and re.match(r"^\d+,", line) and line.count(",") == len(cols) - 1:
        rows.append(line.split(","))
df = pd.DataFrame(rows, columns=cols).apply(pd.to_numeric, errors="coerce")
us = df["infer_us"].to_numpy()
summary = {
    "dataset": a.dataset, "board": meta.get("board"), "cpu_mhz": meta.get("cpu_mhz"),
    "n_nodes": meta.get("n_nodes"), "n_used_features": meta.get("n_used_features"),
    "n_samples": int(len(df)), "matches": int((df["board_pred"] == df["python_pred"]).sum()),
    "classes_in_subset": int(df["true_label"].nunique()),
    "mean_us": float(us.mean()), "median_us": float(np.median(us)), "p95_us": float(np.percentile(us, 95)),
    "p99_us": float(np.percentile(us, 99)), "max_us": float(us.max()),
    "free_ram_bytes": meta.get("free_ram_bytes"), "sketch_bytes": a.sketch_bytes,
    "complete": "#DONE" in text,
}
out = ROOT / "results" / "uno_export" / a.dataset / "uno_summary.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
