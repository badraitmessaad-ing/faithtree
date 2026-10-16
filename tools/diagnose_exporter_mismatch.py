"""
diagnose_exporter_mismatch.py -- WHY does an exporter (emlearn) disagree with scikit-learn on the board?

emlearn's float ("inline") code compares   features[i] <  ROUND6(threshold)   (strict '<', threshold printed with 6 decimals),
scikit-learn's rule (and this framework's export) is   x <= threshold   (threshold kept to float32 precision).
This script re-traces every flashed test vector through the production tree under four rules and compares them with the
predictions the BOARD actually produced (esp32/serial_logs/toolchain_prod_emlearn.txt):

    sklearn      : x <= thr               (reference)
    emlearn-sim  : x <  round6(thr)       (what emlearn's generated code does)
    strict-only  : x <  thr               (effect of the comparison operator alone)
    round-only   : x <= round6(thr)       (effect of the 6-decimal rounding alone)

If emlearn-sim reproduces the board's emlearn predictions, the cause of the disagreement is identified.
Needs no compiler and no raw data.   Usage (from the repository root):
    python tools/diagnose_exporter_mismatch.py --dataset nf_ton_iot_v2 [--variant prod_emlearn]
"""
import argparse, os, re, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="nf_ton_iot_v2"); ap.add_argument("--variant", default="prod_emlearn")
a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
os.environ["ACTIVE_DATASET"] = a.dataset; os.environ["PAPER_PACKAGE_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT / "src")); os.chdir(ROOT / "results")
from eval_helpers import load_artifact_and_metadata          # noqa: E402
import toolchain_export as tc                               # noqa: E402
import paper_utils as pu                                    # noqa: E402

art, meta = load_artifact_and_metadata()
tree = art["pipeline"].named_steps["clf"].tree_
vdir = ROOT / "results" / "esp32_export" / a.dataset / "toolchain_compare" / a.variant / "esp32_dt_test"
txt = (vdir / "esp32_test_vectors.h").read_text()
body = txt[txt.index("TEST_X[TEST_SAMPLE_COUNT][TEST_FEATURE_COUNT] = {"):]; body = body[body.index("{") + 1: body.rindex("};")]
tok = {"NAN": np.nan, "INFINITY": np.inf, "-INFINITY": -np.inf}
rows = [[tok.get(t.strip(), None) if t.strip() in tok else float(t) for t in r.split(",") if t.strip()] for r in re.findall(r"\{([^{}]*)\}", body)]
raw = np.array(rows, dtype=np.float32)
ids = [int(v) for v in re.search(r"TEST_SAMPLE_ID\[TEST_SAMPLE_COUNT\] = \{([^}]*)\}", txt).group(1).split(",")]
x = tc._selected_f32(art, raw)                               # exact float32 preprocessing, as on the device

left, right, feat = tree.children_left, tree.children_right, tree.feature
thr32 = tree.threshold.astype(np.float32)
thr_r6 = np.array([np.float32(float(f"{t:.6f}")) for t in tree.threshold], dtype=np.float32)
leaf_cls = np.argmax(tree.value.reshape(tree.node_count, -1), axis=1)

def trace(rule):
    out = np.empty(len(x), dtype=np.int64)
    for i, xi in enumerate(x):
        n = 0
        while left[n] != -1:
            v, f = xi[feat[n]], feat[n]
            go = (v <= thr32[n]) if rule == "sklearn" else (v < thr_r6[n]) if rule == "emlearn" else (v < thr32[n]) if rule == "strict" else (v <= thr_r6[n])
            n = left[n] if go else right[n]
        out[i] = leaf_cls[n]
    return out

P = {r: trace(r) for r in ("sklearn", "emlearn", "strict", "round")}
log = pu.parse_serial_log(ROOT / "esp32" / "serial_logs" / f"toolchain_{a.variant}.txt")["samples"]
board = log.set_index("sample_id")["esp32_pred"].reindex(ids).to_numpy()
pyref = log.set_index("sample_id")["python_pred"].reindex(ids).to_numpy()
n = len(ids)
print(f"vectors: {n} | board {a.variant} vs Python: {int((board == pyref).sum())}/{n}")
print(f"sanity: our sklearn trace vs the Python predictions in the log: {int((P['sklearn'] == pyref).sum())}/{n}  (must be {n})")
if (P["sklearn"] == pyref).mean() < 0.999:
    sys.exit("STOP: the log does not belong to these test vectors / this model (wrong or stale log, or the variant folder was rebuilt after flashing). "
             "Re-flash the CURRENT variant folder and save a fresh log.")
print(f"\nemlearn-sim (x <  round6(thr)) reproduces the BOARD for {int((P['emlearn'] == board).sum())}/{n} vectors")
print(f"   -> {'MECHANISM CONFIRMED' if (P['emlearn'] == board).all() else 'mechanism only partly explains the board (other differences exist)'}")
mis = board != pyref
print(f"\nof the {int(mis.sum())} board mismatches:")
print(f"   explained by the strict '<' alone        : {int((mis & (P['strict'] != P['sklearn'])).sum())}")
print(f"   explained by the 6-decimal rounding alone: {int((mis & (P['round'] != P['sklearn'])).sum())}")
print(f"   reproduced by both together (emlearn-sim): {int((mis & (P['emlearn'] == board)).sum())}")
