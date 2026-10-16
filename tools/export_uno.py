"""
export_uno.py -- build the Arduino Uno version of a saved model (exact export, flash-resident arrays).

Needs the saved model of the dataset and the file written by notebook 03
(results/esp32_export/<dataset>/python_expected_predictions.csv); no raw CSV is read.

    python run_all.py --mode full --datasets nf_ton_iot_v2_binary --steps 3      # once, if not done yet
    python tools/export_uno.py --dataset nf_ton_iot_v2_binary

Writes results/uno_export/<dataset>/uno_dt_test/ (uno_dt_test.ino, uno_dt_model.h, uno_test_vectors.h)
and checks the C logic against scikit-learn on every row of that CSV (numpy, and compiled C when gcc exists).
A tree that cannot fit in the Uno's 32 KB flash is refused.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--n", type=int, default=200, help="maximum number of test vectors to flash")
a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
os.environ["ACTIVE_DATASET"] = a.dataset; os.environ["PAPER_PACKAGE_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT / "src")); (ROOT / "results").mkdir(exist_ok=True); os.chdir(ROOT / "results")
from eval_helpers import load_artifact_and_metadata      # noqa: E402
import esp32_export as ex                                # noqa: E402
import uno_export as ux                                  # noqa: E402

art, meta = load_artifact_and_metadata()
pipe = art["pipeline"]; cols = list(art["numeric_cols"])
csv = ROOT / "results" / "esp32_export" / a.dataset / "python_expected_predictions.csv"
if not csv.exists():
    sys.exit(f"{csv} not found: run notebook 03 for this dataset first "
             f"(python run_all.py --mode full --datasets {a.dataset} --steps 3)")
df = pd.read_csv(csv)
X = df[cols].astype(np.float32)

out = ROOT / "results" / "uno_export" / a.dataset / "uno_dt_test"
info = ux.export_tree_pipeline_to_uno(art, out)
n_fit, model_bytes = ux.max_vectors_for_flash(info["n_nodes"], info["n_used_features"])
print(f"tree: {info['n_nodes']:,} nodes, {info['n_used_features']} raw features used, model {model_bytes:,} B in flash")
if n_fit < 20:
    sys.exit(f"does not fit: the model alone takes {model_bytes:,} B of the Uno's {ux.UNO_FLASH_BYTES:,} B "
             f"(about {ux.UNO_FLASH_BYTES - ux.UNO_RESERVED_BYTES:,} B are free for model + test vectors)")

# check the Uno C logic on every row of the CSV (numpy re-implementation)
X_used = X.iloc[:, info["used_raw_index"]].to_numpy(np.float32)
py_all = pipe.predict(X)
emu = ux.emulate_uno_predict(info, X_used)
print(f"Uno C logic vs scikit-learn (numpy): {int((emu == py_all).sum())}/{len(py_all)} rows agree")

# choose the flashed vectors: class-balanced, as many as fit
n = min(a.n, n_fit, len(df))
sel = ex.select_validation_samples(df["true_label"].to_numpy(), n, "class_balanced", 42)
ux.create_uno_test_vectors_header(X_used[sel], df["true_label"].to_numpy()[sel], py_all[sel], df["sample_id"].to_numpy()[sel], out)
ino = ux.write_uno_sketch(out)
est = model_bytes + n * (4 * info["n_used_features"] + 6) + ux.UNO_RESERVED_BYTES
print(f"{n} test vectors flashed ({len(np.unique(df['true_label'].to_numpy()[sel]))} classes); "
      f"estimated sketch size {est:,} B of {ux.UNO_FLASH_BYTES:,} B")

host = ux.host_check(out, py_all[sel])
print("compiled C on this PC:", host if host is not None else "no C compiler found (optional)")
(out.parent / "uno_check.json").write_text(json.dumps({
    "dataset": a.dataset, "n_nodes": info["n_nodes"], "n_used_features": info["n_used_features"],
    "model_flash_bytes": model_bytes, "n_flashed": int(n), "estimated_sketch_bytes": int(est),
    "numpy_agreement_rows": int((emu == py_all).sum()), "numpy_rows": int(len(py_all)),
    "raw_threshold_check": info["raw_threshold_check"], "host_c": host}, indent=2, default=str))

print(f"""
================  ON THE ARDUINO UNO  ================
1. Arduino IDE: open  {ino}
   Board: Arduino Uno. Upload. Note the line "Sketch uses ... bytes" from the compile output.
   If the IDE says the sketch is too big, rerun this tool with a smaller --n (e.g. --n 100).
2. Serial Monitor at 115200 baud; wait for #DONE (press RESET if the start was missed).
3. Save the whole output as  {ROOT / 'esp32' / 'serial_logs' / (a.dataset + '_uno_serial.txt')}
   (or: python tools/capture_serial.py --port COM5 --dataset {a.dataset} --tag uno_serial)
4. python tools/analyze_uno_log.py --dataset {a.dataset} --sketch-bytes <number from step 1>
======================================================
""")
