"""
verify_exact_export.py -- checks the exact ESP32 export of a saved model without the raw data.

For the saved model of a dataset it
  1. computes the raw-space thresholds and checks, node by node against the fitted
     scikit-learn pipeline, that the cut value goes left and the next float32 goes right;
  2. builds adversarial inputs that sit exactly on, just above and just below every cut
     (100,000 rows by default), the same rows with NaN / +inf / -inf injected, and uniformly random float32
     bit patterns, and compares scikit-learn, the exact C logic and the legacy C logic on each batch;
  3. if a C compiler is available, compiles the generated header and runs it on the same rows.

Usage (from the repository root):
    python tools/verify_exact_export.py --dataset nf_ton_iot_v2 [--rows 100000]
"""
import argparse, json, os, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="nf_ton_iot_v2"); ap.add_argument("--rows", type=int, default=100_000)
a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
os.environ["ACTIVE_DATASET"] = a.dataset; os.environ["PAPER_PACKAGE_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT / "src")); os.chdir(ROOT / "results")
from eval_helpers import load_artifact_and_metadata      # noqa: E402
import esp32_export as ex                                # noqa: E402

art, meta = load_artifact_and_metadata()
pipe = art["pipeline"]; tree = pipe.named_steps["clf"].tree_
rf, rt = ex.raw_space_thresholds(art); art["_raw_cache"] = (rf, rt)
chk = ex.verify_raw_thresholds(art, rf, rt)
chk["edge_cases"] = ex.verify_edge_cases(art, rf, rt)
chk["monotone"] = ex.verify_monotone(art, rf)
print("per-node check:", {k: v for k, v in chk.items() if k not in ("edge_cases", "monotone")})
print("edge cases (NaN, +/-inf, +/-0, +/-FLT_MAX at every node):", chk["edge_cases"])
print("monotonicity spot check:", chk["monotone"])

rng = np.random.default_rng(0)
internal = np.where(tree.children_left != -1)[0]
fin = internal[np.isfinite(rt[internal])]
med = pipe.named_steps["imputer"].statistics_.astype(np.float32)
B = np.tile(med, (a.rows, 1)).astype(np.float32)
for j in range(B.shape[1]):
    nodes = fin[rf[fin] == j]
    if len(nodes) == 0:
        continue
    pick = rt[rng.choice(nodes, size=a.rows)]
    step = rng.integers(-1, 2, size=a.rows)
    up = np.nextafter(pick, np.float32(np.inf), dtype=np.float32)
    dn = np.nextafter(pick, np.float32(-np.inf), dtype=np.float32)
    B[:, j] = np.where(step > 0, up, np.where(step < 0, dn, pick))
# batch 2: the same rows with about 10 % of the values replaced by NaN, +inf or -inf
N = B.copy()
mask = rng.random(N.shape) < 0.10
N[mask] = rng.choice(np.array([np.nan, np.inf, -np.inf], dtype=np.float32), size=int(mask.sum()))
# batch 3: uniformly random float32 bit patterns (any value, including huge numbers, subnormals, NaN and inf)
R = rng.integers(0, 2**32, size=(a.rows, B.shape[1]), dtype=np.uint64).astype(np.uint32).view(np.float32)

d = Path(tempfile.mkdtemp(dir=ex._safe_tmp()))
ex.export_tree_pipeline_to_c(art, d)
res = {"dataset": a.dataset, "model": meta["model_path"], "per_node_check": chk, "rows_per_batch": int(a.rows)}
for name, M in [("cuts_and_neighbours", B), ("with_nan_and_inf", N), ("random_float32_bit_patterns", R)]:
    S = pd.DataFrame(M, columns=art["numeric_cols"])
    sk = pipe.predict(S)
    full = ex.host_c_check_full(d, art, S, sk)
    res[name] = {"exact_c_logic_mismatches": int((ex.emulate_c_predict(art, S) != sk).sum()),
                 "legacy_c_logic_mismatches": int((ex.emulate_c_predict_legacy(art, S) != sk).sum()),
                 "compiled_exact_header": "no C compiler found" if full is None else {k: v for k, v in full.items() if k != "c_pred"}}
print(json.dumps(res, indent=2, default=str))
out = ROOT / "results" / "esp32_export" / a.dataset / "exact_export_verification.json"
out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=2, default=str))
print("saved", out)
