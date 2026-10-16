"""
run_settings.py -- ONE place that controls how heavy a run is.

Set the environment variable PAPER_MODE (or edit DEFAULT_MODE):
  full  : the settings used for the paper (hours on the big datasets)
  quick : ~10x cheaper search (good first run / sanity check on the real data)
  smoke : tiny budgets, for testing the whole pipeline in minutes (used with the synthetic data)
"""
import os

DEFAULT_MODE = "full"
MODE = os.environ.get("PAPER_MODE", DEFAULT_MODE).lower()

_PRESETS = {
    "full": dict(
        main_overrides={},                       # 45/30/20 budgets, K in 20..40, 300k sample
        n_repeat_seeds=5,                        # multi-seed robustness
        baseline_sample_limit=500_000,
        ablation_sample_limit=500_000,
        ablation_ks=[10, 20, 25, 30, 35, 40],
        opt_cmp_repeats=5, opt_cmp_sample_limit=300_000, opt_cmp_budget=95,      # same budget as the main search (45/30/20 = 95 evaluations, 300k search sample)
        opt_cmp_refit_limit=0, opt_cmp_eval_limit=0,                             # 0 = ALL training rows / the FULL validation and test splits (paper scale)
        esp32_n_samples=5000,
        fp_train_limit=1_000_000, fp_pool=1_500_000, fp_eval_limit=1_000_000, fp_repeats=5,
        fixed_k_rows=None,
        hr_train_rows=3_000_000, hr_eval_rows=1_000_000, hr_repeats=3, hr_forest_sizes=[5, 10, 20], hr_boost_rounds=[10, 20, 40],
        tc_n_samples=2000, tc_train_rows=3_000_000, tc_eval_rows=1_000_000, tc_forest_trees=10,
        gen_train_rows=3_000_000, gen_eval_rows=1_000_000, gen_sample_rows=6_000_000, gen_host_repeats=5,
    ),
    "quick": dict(
        main_overrides=dict(ks=[25, 35], silverback_budget=20, dolphin_budget=12,
                            prune_budget=8, sample_limit=100_000),
        n_repeat_seeds=3,
        baseline_sample_limit=200_000,
        ablation_sample_limit=200_000,
        ablation_ks=[10, 20, 35],
        opt_cmp_repeats=5, opt_cmp_sample_limit=50_000, opt_cmp_budget=40,
        opt_cmp_refit_limit=300_000, opt_cmp_eval_limit=300_000,
        esp32_n_samples=5000,
        fp_train_limit=300_000, fp_pool=450_000, fp_eval_limit=300_000, fp_repeats=3,
        fixed_k_rows=3_000_000,
        hr_train_rows=1_000_000, hr_eval_rows=500_000, hr_repeats=1, hr_forest_sizes=[5, 10], hr_boost_rounds=[10, 20],
        tc_n_samples=2000, tc_train_rows=1_000_000, tc_eval_rows=500_000, tc_forest_trees=10,
        gen_train_rows=1_000_000, gen_eval_rows=500_000, gen_sample_rows=4_000_000, gen_host_repeats=5,
    ),
    "smoke": dict(
        main_overrides=dict(ks=[10, 15], silverback_budget=8, dolphin_budget=5,
                            prune_budget=4, sample_limit=15_000),
        n_repeat_seeds=2,
        baseline_sample_limit=20_000,
        ablation_sample_limit=20_000,
        ablation_ks=[10, 15],
        opt_cmp_repeats=2, opt_cmp_sample_limit=12_000, opt_cmp_budget=16,
        opt_cmp_refit_limit=15_000, opt_cmp_eval_limit=8_000,
        esp32_n_samples=500,
        fp_train_limit=12_000, fp_pool=18_000, fp_eval_limit=8_000, fp_repeats=2,
        fixed_k_rows=None,
        hr_train_rows=12_000, hr_eval_rows=6_000, hr_repeats=1, hr_forest_sizes=[3, 5], hr_boost_rounds=[5],
        tc_n_samples=500, tc_train_rows=12_000, tc_eval_rows=6_000, tc_forest_trees=3,
        gen_train_rows=12_000, gen_eval_rows=6_000, gen_sample_rows=24_000, gen_host_repeats=2,
    ),
}
if MODE not in _PRESETS:
    raise ValueError(f"PAPER_MODE must be one of {list(_PRESETS)}, got {MODE!r}")

SETTINGS = _PRESETS[MODE]
MAIN_OVERRIDES = SETTINGS["main_overrides"]
N_REPEAT_SEEDS_CFG = SETTINGS["n_repeat_seeds"]
BASELINE_SAMPLE_LIMIT_CFG = SETTINGS["baseline_sample_limit"]
ABLATION_SAMPLE_LIMIT_CFG = SETTINGS["ablation_sample_limit"]
ABLATION_KS_CFG = SETTINGS["ablation_ks"]
# --- notebook 04 (optimizer comparison).  Env vars override the mode's defaults, e.g.  set OPT_CMP_REPEATS=10
OPT_CMP_REPEATS = int(os.environ.get("OPT_CMP_REPEATS", SETTINGS["opt_cmp_repeats"]))
OPT_CMP_BUDGET = int(os.environ.get("OPT_CMP_BUDGET", SETTINGS["opt_cmp_budget"]))
OPT_CMP_SAMPLE_LIMIT = int(os.environ.get("OPT_CMP_SAMPLE_LIMIT", SETTINGS["opt_cmp_sample_limit"]))   # rows the optimisers search on
OPT_CMP_REFIT_LIMIT = int(os.environ.get("OPT_CMP_REFIT_LIMIT", SETTINGS["opt_cmp_refit_limit"]))        # rows each winner is refitted on; 0 = ALL training rows
OPT_CMP_EVAL_LIMIT = int(os.environ.get("OPT_CMP_EVAL_LIMIT", SETTINGS["opt_cmp_eval_limit"]))          # rows of the validation / test split used for scoring; 0 = all
# --- notebook 07 (feature-policy / leakage ablation).  Env var FP_REPEATS overrides the mode's default.
FP_TRAIN_LIMIT = int(os.environ.get("FP_TRAIN_LIMIT", SETTINGS["fp_train_limit"]))   # rows each tree is trained on (env FP_TRAIN_LIMIT overrides)
FP_POOL = int(os.environ.get("FP_POOL", int(FP_TRAIN_LIMIT * 1.5) if "FP_TRAIN_LIMIT" in os.environ else SETTINGS["fp_pool"]))  # train rows loaded once
FP_EVAL_LIMIT = SETTINGS["fp_eval_limit"]       # rows of the validation / test split used for scoring
FP_REPEATS = int(os.environ.get("FP_REPEATS", SETTINGS["fp_repeats"]))
# --- notebook 08 (fixed-configuration training).  Rows used to compare the K candidates (None = all training rows).
FIXED_K_ROWS = int(os.environ["FIXED_K_ROWS"]) if "FIXED_K_ROWS" in os.environ else SETTINGS["fixed_k_rows"]
# --- notebook 10 (headroom analysis: ensembles + derived features vs the single tree)
HR_TRAIN_ROWS = int(os.environ.get("HR_TRAIN_ROWS", SETTINGS["hr_train_rows"]))
HR_EVAL_ROWS = int(os.environ.get("HR_EVAL_ROWS", SETTINGS["hr_eval_rows"]))
HR_REPEATS = int(os.environ.get("HR_REPEATS", SETTINGS["hr_repeats"]))
HR_FOREST_SIZES = [int(v) for v in os.environ.get("HR_FOREST_SIZES", ",".join(map(str, SETTINGS["hr_forest_sizes"]))).split(",")]
HR_BOOST_ROUNDS = [int(v) for v in os.environ.get("HR_BOOST_ROUNDS", ",".join(map(str, SETTINGS["hr_boost_rounds"]))).split(",")]
HR_MODEL_BUDGET_KB = float(os.environ.get("HR_MODEL_BUDGET_KB", 800))   # flash available for the MODEL (1.25 MB app partition minus ~0.3 MB core/firmware; an assumption; check it on the target build)
# --- notebook 11 (toolchain / model-family comparison on the board)
TC_N_SAMPLES = int(os.environ.get("TC_N_SAMPLES", SETTINGS["tc_n_samples"]))        # test vectors flashed with EVERY variant (keep modest: forests are big)
TC_TRAIN_ROWS = int(os.environ.get("TC_TRAIN_ROWS", SETTINGS["tc_train_rows"]))     # rows for the model-family comparison (tree / depth-5 / forest trained on the SAME rows)
TC_EVAL_ROWS = int(os.environ.get("TC_EVAL_ROWS", SETTINGS["tc_eval_rows"]))        # test rows used to compute each variant's MCC
TC_FOREST_TREES = int(os.environ.get("TC_FOREST_TREES", SETTINGS["tc_forest_trees"]))
# --- notebooks 12 / 14 (generalization: unseen-host split, time split, cross-dataset)
GEN_TRAIN_ROWS = int(os.environ.get("GEN_TRAIN_ROWS", SETTINGS["gen_train_rows"]))
GEN_EVAL_ROWS = int(os.environ.get("GEN_EVAL_ROWS", SETTINGS["gen_eval_rows"]))
GEN_SAMPLE_ROWS = int(os.environ.get("GEN_SAMPLE_ROWS", SETTINGS["gen_sample_rows"]))      # rows sampled once for notebook 12 (uniform; all splits are done in memory)
GEN_HOST_REPEATS = int(os.environ.get("GEN_HOST_REPEATS", SETTINGS["gen_host_repeats"]))   # different random bucket->partition assignments
GEN_FINGERPRINT_COLS = [c for c in os.environ.get("GEN_FINGERPRINT_COLS", "MIN_TTL,MAX_TTL,TCP_WIN_MAX_IN,TCP_WIN_MAX_OUT").split(",") if c]
ESP32_N_SAMPLES = int(os.environ.get("ESP32_N_SAMPLES", SETTINGS["esp32_n_samples"]))
# "class_balanced" puts every class (incl. rare ones) on the board; "first_n" = first rows only.
ESP32_SAMPLE_MODE = os.environ.get("ESP32_SAMPLE_MODE", "class_balanced")

print(f"[run_settings] PAPER_MODE = {MODE!r}")
