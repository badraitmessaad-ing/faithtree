"""
run_all.py -- run the FaithTree notebooks in order, for one or several datasets.

Examples
  python run_all.py --check                                   # check CSVs and packages, run nothing
  python run_all.py --mode smoke --datasets nf_unsw_nb15_v3   # minutes, tiny budgets
  python run_all.py --mode quick                              # all three datasets, about 10x cheaper search
  python run_all.py --mode full                               # paper settings (hours)
  python run_all.py --steps 8 2 3 5 6                         # fixed-configuration model, evaluation, export, figures
  python run_all.py --steps 5 6                               # after flashing the board: hardware analysis + figures
  python run_all.py --binary --steps 8 2                      # binary (benign vs attack) profiles; then --steps 9

Steps (step number = notebook number)
   1  search and train (Silverback -> Dolphin -> pruning)          per dataset
   8  train a fixed, validation-selected configuration (alt. to 1)  per dataset
   2  evaluation, same-split baselines, ablations                    per dataset
   3  ESP32 export (exact + legacy), test vectors, host checks       per dataset
   4  optimizer comparison at equal budget (slow)                    per dataset
   7  feature-policy ablation (ports, identifier patterns)           per dataset
  10  ceiling analysis: ensembles and derived features               per dataset
  11  exporter and model-family comparison (emlearn, micromlgen)     per dataset
  12  generalisation: unseen-host and time splits                    per dataset
   5  ESP32 hardware analysis from the serial logs                   once
   6  paper figures and tables                                       once
   9  binary-task results                                            once
  14  cross-dataset test (NF-ToN-IoT-v2 <-> NF-UNSW-NB15-v3)          once
  15  analysis of the exporter-comparison board logs                 once

Default steps: 1 2 3 5 6.  Executed copies of the notebooks are written to notebooks/executed/.
Notebook 04 knobs (environment variables): OPT_CMP_REPEATS, OPT_CMP_BUDGET, OPT_CMP_NODE_CAP, OPT_CMP_MCC_TOL.
"""
import argparse, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ["nf_ton_iot_v2", "cic_ton_iot", "nf_unsw_nb15_v3"]
CSV = {"nf_ton_iot_v2": "NF-ToN-IoT-v2.csv", "cic_ton_iot": "CIC-ToN-IoT.csv", "nf_unsw_nb15_v3": "NF-UNSW-NB15-v3.csv"}
BINARY = [d + "_binary" for d in DATASETS]                       # same CSV files, benign-vs-attack task
CSV.update({d + "_binary": CSV[d] for d in DATASETS})
NB = {1: "01_train_and_optimize.ipynb", 2: "02_evaluation_baselines_ablation.ipynb", 3: "03_esp32_export.ipynb",
      4: "04_optimizer_comparison.ipynb", 5: "05_esp32_hardware_analysis.ipynb", 6: "06_paper_figures_and_tables.ipynb",
      7: "07_feature_policy_ablation.ipynb", 8: "08_train_fixed_config.ipynb", 9: "09_binary_results.ipynb",
      10: "10_headroom_analysis.ipynb", 11: "11_toolchain_comparison.ipynb", 12: "12_generalization_splits.ipynb",
      14: "14_cross_dataset.ipynb", 15: "15_toolchain_log_analysis.ipynb"}
PER_DATASET = {1, 2, 3, 4, 7, 8, 10, 11, 12}


def check(datasets):
    ok = True
    data_dir = Path(os.environ.get("PAPER_DATA_DIR", ROOT / "data"))
    print("data folder:", data_dir)
    for d in datasets:
        p = data_dir / CSV[d]
        s = f"{p.stat().st_size/1e6:,.0f} MB" if p.exists() else "MISSING"
        print(f"  {d:18s} {CSV[d]:22s} {s}"); ok &= p.exists()
    for m in ["numpy", "pandas", "sklearn", "matplotlib", "nbformat", "nbclient", "ipykernel"]:
        try: __import__(m); print(f"  package {m:12s} ok")
        except Exception: print(f"  package {m:12s} MISSING  ->  pip install -r requirements.txt"); ok = False
    for m in ["xgboost", "lightgbm", "optuna", "seaborn"]:
        try: __import__(m); print(f"  optional {m:11s} ok")
        except Exception: print(f"  optional {m:11s} not installed (that baseline/method is skipped)")
    return ok


def run_nb(name, env, outdir):
    import nbformat
    from nbclient import NotebookClient
    path = ROOT / "notebooks" / name
    nb = nbformat.read(str(path), as_version=4)
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    t = time.time()
    try:
        NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
        status = "OK"
    except Exception as e:
        status = "FAILED"
        print("   ", str(e)[-1800:])
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    outdir.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(outdir / name))
    print(f"   {status}  {time.time()-t:6.0f}s  {name}")
    return status == "OK"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS + BINARY)
    ap.add_argument("--binary", action="store_true", help="shortcut: run the three BINARY (benign vs attack) profiles")
    ap.add_argument("--steps", nargs="+", type=int, default=[1, 2, 3, 5, 6], choices=sorted(NB))
    ap.add_argument("--mode", default=os.environ.get("PAPER_MODE", "full"), choices=["smoke", "quick", "full"])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed notebook")
    a = ap.parse_args()
    if a.binary: a.datasets = BINARY

    needs_data = bool(set(a.steps) & PER_DATASET)
    check_list = list(a.datasets) if needs_data else []
    if 14 in a.steps:                      # cross-dataset test needs both NetFlow CSVs
        check_list += [d for d in ("nf_ton_iot_v2", "nf_unsw_nb15_v3") if d not in check_list]
    ok = check(check_list)
    if a.check:
        sys.exit(0 if ok else 1)
    if not ok:
        sys.exit("fix the problems above first (or use --steps 5 6 which need no raw data)")

    env_common = {"PAPER_MODE": a.mode}
    print(f"\nmode={a.mode}  datasets={a.datasets}  steps={a.steps}\n")
    for ds in a.datasets:
        for st in [s for s in a.steps if s in PER_DATASET]:
            print(f"[{ds}] step {st}")
            if not run_nb(NB[st], {**env_common, "ACTIVE_DATASET": ds}, ROOT / "notebooks" / "executed" / ds) and not a.keep_going:
                sys.exit(f"stopped: {NB[st]} failed for {ds}")
    for st in [s for s in a.steps if s not in PER_DATASET]:
        print(f"[all] step {st}")
        if not run_nb(NB[st], env_common, ROOT / "notebooks" / "executed") and not a.keep_going:
            sys.exit(f"stopped: {NB[st]} failed")
    print("\nDone.  Figures -> paper/figures   Tables -> paper/tables")


if __name__ == "__main__":
    main()
