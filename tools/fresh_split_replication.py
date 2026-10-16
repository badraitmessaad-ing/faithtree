"""
fresh_split_replication.py -- repeats the WHOLE model-selection procedure on new, never-seen splits.

Why: the seed-42 test split was printed during development, so the paper's test MCC could in
principle be optimistic. This script checks that by re-running selection on fresh splits:

  for every new seed
    1. stratified 64/16/20 split (train / validation / test) with that seed
    2. candidates, all scored on VALIDATION only
         a) the paper's fixed configuration, K chosen from the K grid
         b) restricted random search: BUDGET configurations drawn from the restricted space x K grid
       stage 1: every candidate is fitted on a stratified subsample of the training split
       stage 2: the best fixed-config K and the TOP_N search candidates are refitted on the full
                training split and scored on the full validation split
       selection: best validation MCC among finalists that fit the export format (<= 32,767 nodes);
                  ties -> smaller tree
    3. the winner is refitted on train + validation and the test split is scored ONCE

All rules are written to protocol.json before the first fit. Nothing is tuned on a test split.

Usage (from the repository root):
    python tools/fresh_split_replication.py --dataset nf_ton_iot_v2
    python tools/fresh_split_replication.py --dataset nf_ton_iot_v2 --seeds 7 101 2027 --budget 95 --search-rows 300000

Outputs: results/fresh_split_replication/<dataset>/ protocol.json, search_log.csv, finalists.csv, runs.csv, summary.json
"""
import argparse, json, os, sys, time, gc
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="nf_ton_iot_v2")
ap.add_argument("--seeds", type=int, nargs="+", default=[7, 101, 2027],
                help="new split seeds; must differ from 42 (development) and 142-144 (multi-seed check)")
ap.add_argument("--budget", type=int, default=95, help="random-search configurations per seed (Section IV-H used 95)")
ap.add_argument("--search-rows", type=int, default=300_000, help="training rows used to score each candidate in stage 1")
ap.add_argument("--top-n", type=int, default=3, help="search candidates refitted on the full training split in stage 2")
ap.add_argument("--fixed-config", default="configs/nf_ton_iot_v2_fixed_params.json")
ap.add_argument("--reported-test-mcc", type=float, default=0.9722, help="the paper's seed-42 test MCC, for comparison only")
a = ap.parse_args()

ROOT = Path(__file__).resolve().parents[1]
os.environ["ACTIVE_DATASET"] = a.dataset; os.environ["PAPER_PACKAGE_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT / "src")); (ROOT / "results").mkdir(exist_ok=True); os.chdir(ROOT / "results")
from framework import (build_pipeline, load_and_prepare_dataframe, stratified_sample,      # noqa: E402
                       DEFAULT_DATASET_PATH, TEST_SIZE, VAL_SIZE, SEED, _json_default)
from sklearn.model_selection import train_test_split                                         # noqa: E402
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score                     # noqa: E402

BAD = {SEED} | {SEED + 100 + i for i in range(10)}
if BAD & set(a.seeds):
    sys.exit(f"seeds {sorted(BAD & set(a.seeds))} were already used (development / multi-seed check); choose new ones")

# ------------------------------------------------------------------ pre-registered rules
K_GRID = [20, 25, 30, 32, 34, 36]
RESTRICTED_SPACE = {                                     # Table V, restricted space
    "criterion": ["gini", "entropy"],
    "max_depth": [8, 10, 12, 14, 16, 18, 20, 24, 28, 32, None],
    "min_samples_split": [2, 5, 10, 20, 40],
    "min_samples_leaf": [1, 2, 4, 8, 16, 32, 64],
    "ccp_alpha": [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
}
FIXED_DEFAULTS = {"splitter": "best", "min_weight_fraction_leaf": 0.0, "max_features": None, "class_weight": None}
MAX_NODES = 32_767                                       # export format limit (int16 child indices)
# the paper's fixed configuration (validation-selected in the June 2026 search); read from the config file when present
PAPER_FIXED_PARAMS = {"criterion": "gini", "splitter": "best", "max_depth": 20, "min_samples_split": 10, "min_samples_leaf": 8,
                      "min_weight_fraction_leaf": 0.0, "max_features": None, "class_weight": None, "ccp_alpha": 1e-06}
pf = Path(a.fixed_config); pf = pf if pf.is_absolute() else ROOT / pf
if pf.exists():
    FIXED_PARAMS = dict(json.load(open(pf, encoding="utf-8"))["best_params"])
else:
    print(f"note: {pf} not found; using the paper's fixed configuration built into this script")
    FIXED_PARAMS = dict(PAPER_FIXED_PARAMS); pf = Path("built-in (paper fixed configuration)")

OUT = ROOT / "results" / "fresh_split_replication" / a.dataset; OUT.mkdir(parents=True, exist_ok=True)
protocol = {
    "written_at": datetime.now().isoformat(timespec="seconds"),
    "purpose": "repeat the full selection procedure on fresh splits; one test evaluation per seed",
    "dataset": a.dataset, "seeds": a.seeds, "split": {"test": TEST_SIZE, "validation_of_remainder": VAL_SIZE, "stratified": True},
    "features": "leakage-safe profile of the dataset (same columns as the paper)",
    "candidates": {"fixed_config": FIXED_PARAMS, "fixed_config_source": pf.name, "k_grid": K_GRID,
                   "random_search_space": RESTRICTED_SPACE, "random_search_budget": a.budget, "constant_params": FIXED_DEFAULTS},
    "stage1": f"each candidate fitted on a stratified subsample of {a.search_rows:,} training rows, scored by validation MCC",
    "stage2": f"best fixed-config K and the top {a.top_n} search candidates refitted on the full training split, scored on the full validation split",
    "selection": f"best validation MCC among finalists with <= {MAX_NODES:,} nodes; ties -> fewer nodes",
    "final": "winner refitted on train + validation; test split scored once",
    "reported_seed42_test_mcc": a.reported_test_mcc,
}
json.dump(protocol, open(OUT / "protocol.json", "w", encoding="utf-8"), indent=2, default=_json_default)
print("protocol written:", OUT / "protocol.json")


def score(pipe, X, y):
    p = pipe.predict(X); t = pipe.named_steps["clf"].tree_
    return {"mcc": matthews_corrcoef(y, p), "macro_f1": f1_score(y, p, average="macro", zero_division=0),
            "accuracy": accuracy_score(y, p), "nodes": int(t.node_count), "depth": int(t.max_depth)}


def fit_score(params, k, seed, Xtr, ytr, Xv, yv):
    pipe = build_pipeline({**FIXED_DEFAULTS, **params}, min(k, Xtr.shape[1]), seed)
    t0 = time.perf_counter(); pipe.fit(Xtr, ytr)
    s = score(pipe, Xv, yv); s["fit_seconds"] = round(time.perf_counter() - t0, 1)
    del pipe; gc.collect(); return s


# ------------------------------------------------------------------ data (loaded once)
t0 = time.perf_counter()
X, y, le, numeric_cols, shape = load_and_prepare_dataframe(DEFAULT_DATASET_PATH)
print(f"{a.dataset}: {len(X):,} rows, {len(numeric_cols)} features, {len(le.classes_)} classes ({time.perf_counter() - t0:.0f}s)")
idx_all = np.arange(len(X))

search_log, finalists_log, runs = [], [], []
for seed in a.seeds:
    print(f"\n================ seed {seed} ================")
    tv, te = train_test_split(idx_all, test_size=TEST_SIZE, stratify=y, random_state=seed)
    tr, va = train_test_split(tv, test_size=VAL_SIZE, stratify=y[tv], random_state=seed)
    X_tr, y_tr, X_va, y_va = X.iloc[tr], y[tr], X.iloc[va], y[va]
    X_s, y_s = stratified_sample(X_tr, y_tr, a.search_rows, seed)
    print(f"train {len(tr):,} | validation {len(va):,} | test {len(te):,} (test untouched) | stage-1 rows {len(X_s):,}")

    # stage 1a: fixed configuration, every K
    cands = []
    for k in K_GRID:
        s = fit_score(FIXED_PARAMS, k, seed, X_s, y_s, X_va, y_va)
        cands.append({"seed": seed, "source": "fixed config", "k": k, "params": FIXED_PARAMS, **s})
        print(f"  fixed config K={k:>2}: val MCC {s['mcc']:.4f}  nodes {s['nodes']:,}")
    # stage 1b: restricted random search over space x K grid
    rng = np.random.default_rng(seed); seen = set(); n = 0; tries = 0
    while n < a.budget and tries < 50 * a.budget:
        tries += 1
        p = {name: vals[rng.integers(len(vals))] for name, vals in RESTRICTED_SPACE.items()}
        k = K_GRID[rng.integers(len(K_GRID))]
        key = json.dumps([p, k], sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key); n += 1
        s = fit_score(p, k, seed, X_s, y_s, X_va, y_va)
        cands.append({"seed": seed, "source": "random search", "k": k, "params": p, **s})
        if n % 10 == 0 or n == a.budget:
            best = max((c for c in cands if c["source"] == "random search"), key=lambda c: c["mcc"])
            print(f"  random search {n}/{a.budget}: best val MCC so far {best['mcc']:.4f}")
    search_log += [{**{k_: v for k_, v in c.items() if k_ != "params"}, "params": json.dumps(c["params"], default=str)} for c in cands]

    # stage 2: finalists on the full training split
    fixed_best = max((c for c in cands if c["source"] == "fixed config"), key=lambda c: (round(c["mcc"], 6), -c["nodes"]))
    top = sorted((c for c in cands if c["source"] == "random search"), key=lambda c: (-round(c["mcc"], 6), c["nodes"]))[:a.top_n]
    finals = []
    for c in [fixed_best] + top:
        s = fit_score(c["params"], c["k"], seed, X_tr, y_tr, X_va, y_va)
        f = {"seed": seed, "source": c["source"], "k": c["k"], "params": c["params"], "stage1_val_mcc": c["mcc"], **{f"val_{k_}": v for k_, v in s.items()}}
        finals.append(f)
        print(f"  finalist [{c['source']}, K={c['k']}]: full-train val MCC {s['mcc']:.4f}  nodes {s['nodes']:,}  ({s['fit_seconds']}s)")
    finalists_log += [{**{k_: v for k_, v in f.items() if k_ != "params"}, "params": json.dumps(f["params"], default=str)} for f in finals]
    ok = [f for f in finals if f["val_nodes"] <= MAX_NODES] or finals
    win = max(ok, key=lambda f: (round(f["val_mcc"], 6), -f["val_nodes"]))
    print(f"  selected by validation: {win['source']}, K={win['k']}, val MCC {win['val_mcc']:.4f}")

    # stage 3: refit on train + validation, test once
    X_fin = pd.concat([X_tr, X_va]); y_fin = np.concatenate([y_tr, y_va])
    del X_s, y_s; gc.collect()
    pipe = build_pipeline({**FIXED_DEFAULTS, **win["params"]}, min(win["k"], X.shape[1]), seed)
    pipe.fit(X_fin, y_fin)
    del X_fin, y_fin, X_tr, X_va; gc.collect()
    ts = score(pipe, X.iloc[te], y[te])
    print(f"  TEST (once): MCC {ts['mcc']:.4f}  macro-F1 {ts['macro_f1']:.4f}  nodes {ts['nodes']:,}  depth {ts['depth']}")
    runs.append({"seed": seed, "winner": win["source"], "k": win["k"], "params": json.dumps(win["params"], default=str),
                 "val_mcc": win["val_mcc"], "val_macro_f1": win["val_macro_f1"],
                 "test_mcc": ts["mcc"], "test_macro_f1": ts["macro_f1"], "test_accuracy": ts["accuracy"],
                 "nodes": ts["nodes"], "depth": ts["depth"], "exportable": ts["nodes"] <= MAX_NODES})
    del pipe; gc.collect()
    # save after every seed so a crash keeps finished seeds
    pd.DataFrame(search_log).to_csv(OUT / "search_log.csv", index=False)
    pd.DataFrame(finalists_log).to_csv(OUT / "finalists.csv", index=False)
    pd.DataFrame(runs).to_csv(OUT / "runs.csv", index=False)

R = pd.DataFrame(runs)
summary = {"seeds": a.seeds, "test_mcc_mean": R.test_mcc.mean(), "test_mcc_sd": R.test_mcc.std(ddof=1) if len(R) > 1 else None,
           "test_macro_f1_mean": R.test_macro_f1.mean(), "val_mcc_mean": R.val_mcc.mean(),
           "val_minus_test_mean": (R.val_mcc - R.test_mcc).mean(), "winners": R.winner.value_counts().to_dict(),
           "reported_seed42_test_mcc": a.reported_test_mcc,
           "difference_reported_minus_fresh": a.reported_test_mcc - R.test_mcc.mean()}
json.dump(summary, open(OUT / "summary.json", "w", encoding="utf-8"), indent=2, default=_json_default)
print("\n================ summary ================")
print(R[["seed", "winner", "k", "val_mcc", "test_mcc", "test_macro_f1", "nodes", "depth"]].round(4).to_string(index=False))
sd = f" ± {summary['test_mcc_sd']:.4f}" if summary["test_mcc_sd"] is not None else ""
print(f"\nfresh-split test MCC: {summary['test_mcc_mean']:.4f}{sd}  (paper, seed 42: {a.reported_test_mcc:.4f}; "
      f"difference {summary['difference_reported_minus_fresh']:+.4f})")
print("saved in", OUT)
