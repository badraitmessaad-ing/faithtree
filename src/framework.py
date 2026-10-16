"""
framework.py -- core of the FaithTree pipeline.

Contains the dataset registry, the leakage-aware preprocessing pipeline, the
metrics, the cross-validated fitness, the three-stage tree search
(Silverback -> Dolphin -> pruning) and `main()`, which runs the search,
selects (K, hyper-parameters) by validation MCC, refits on train+validation
and evaluates the test split once.

Paths
  * dataset CSVs default to <repo>/data/<file>.csv and can be overridden with
    the NF_TON_IOT_V2_CSV / CIC_TON_IOT_CSV / NF_UNSW_NB15_V3_CSV environment
    variables (or PAPER_DATA_DIR for the whole folder);
  * outputs are written below the current working directory (the notebooks
    switch to <repo>/results before importing this module).
"""
import os
from pathlib import Path

PACKAGE_ROOT = Path(os.environ.get("PAPER_PACKAGE_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.environ.get("PAPER_DATA_DIR", PACKAGE_ROOT / "data"))

# =============================================================================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import copy
import json
import math
import pickle
import random
import time
import gc
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

# ---------------------------------------------------------------------------
# Dataset registry. This is what makes the code a framework rather than a
# single-dataset script. Every downstream cell (search, baselines, plots,
# ESP32 export) reads DEFAULT_DATASET_PATH / TARGET_COL / *_IF_PRESENT below;
# none of them ever reference a dataset name directly, and the feature count,
# class count, and class names are all derived from the loaded data, not
# hardcoded. To run the framework on a new dataset: add a profile here, then
# set ACTIVE_DATASET to its key. Nothing else needs to change.
# ---------------------------------------------------------------------------
DATASET_PROFILES: Dict[str, Dict[str, Any]] = {
    "nf_ton_iot_v2": {
        "description": "NF-ToN-IoT-v2 (NetFlow v2, 9 attack categories + benign)",
        "path": os.environ.get(
            "NF_TON_IOT_V2_CSV",
            str(DATA_DIR / "NF-ToN-IoT-v2.csv"),
        ),
        "target_col": "Attack",
        "drop_cols": ["Label"],
        "ip_cols": ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"],
    },
    "cic_ton_iot": {
        # CICFlowMeter format of the SAME underlying ToN-IoT traffic as
        # nf_ton_iot_v2 above, but a different feature extractor (~83
        # CICFlowMeter features vs. 43 NetFlow-v2 features) and different
        # column names. This is the strongest possible second dataset for a
        # framework claim, because it proves the pipeline is *feature-set*
        # agnostic, not just dataset-agnostic within one flow format.
        #
        # Header confirmed directly from the file (2026-09-12):
        # target_col="Attack" (multi-class), "Label" is the separate binary
        # column, and the identifier columns below match exactly. The
        # pattern-based safety net (LEAKY_COLUMN_PATTERNS) also independently
        # catches these, so a typo here won't silently leak an identifier.
        "description": "CIC-ToN-IoT (CICFlowMeter format, ~83 features) — second profile / feature-set generality check",
        "path": os.environ.get(
            "CIC_TON_IOT_CSV",
            str(DATA_DIR / "CIC-ToN-IoT.csv"),
        ),
        "target_col": "Attack",
        "drop_cols": ["Label"],
        "ip_cols": ["Flow ID", "Src IP", "Dst IP", "Src Port", "Dst Port", "Timestamp"],
    },
    "nf_unsw_nb15_v3": {
        # A genuinely different network environment/testbed than ToN-IoT
        # (UNSW-NB15, different attack taxonomy: Fuzzers/Exploits/Generic/
        # Reconnaissance/Shellcode/Worms/Analysis/Backdoor/DoS). This is
        # what actually tests cross-environment generalization, unlike the
        # nf_ton_iot_v2 / cic_ton_iot pair above (same traffic, two feature
        # extractors). NOTE: this is NetFlow v3 (53 features), not v2 (43
        # features) -- confirmed from the real file header (2026-09-13).
        # target_col="Attack", "Label" is the separate binary column.
        # FLOW_START_MILLISECONDS / FLOW_END_MILLISECONDS are absolute
        # timestamps (not caught by the "timestamp" pattern match since
        # they don't contain that substring) so they're listed explicitly
        # below alongside the IP/port identifier columns.
        "description": "NF-UNSW-NB15-v3 (NetFlow v3, 53 features, UNSW-NB15 testbed) — third profile / cross-environment generality check",
        "path": os.environ.get(
            "NF_UNSW_NB15_V3_CSV",
            str(DATA_DIR / "NF-UNSW-NB15-v3.csv"),
        ),
        "target_col": "Attack",
        "drop_cols": ["Label"],
        "ip_cols": [
            "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT",
            "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS",
        ],
    },
    # Add further profiles here using the same four fields:
    # path, target_col, drop_cols, ip_cols.
}

# ---------------------------------------------------------------------------
# BINARY profiles (benign vs attack), derived from every profile above:
#   <name>_binary : target = the 0/1 "Label" column, the multi-class "Attack" column is dropped
#                   from the features, and kept as `attack_col` ONLY so that the evaluation can
#                   report the detection rate per attack type (never used for training).
# Same CSV file, same feature policy (IPs / ports / identifiers excluded), separate result folders.
# ---------------------------------------------------------------------------
for _k in list(DATASET_PROFILES):
    _p = DATASET_PROFILES[_k]
    DATASET_PROFILES[_k + "_binary"] = {
        **_p,
        "description": _p["description"] + "  --  BINARY task (Label: 0 = benign, 1 = attack)",
        "target_col": "Label",
        "drop_cols": ["Attack"],
        "attack_col": _p["target_col"],
        "binary_label_names": ("Benign", "Malicious"),
    }

# <<< Set which registered profile is active for this run. >>>
ACTIVE_DATASET = os.environ.get("ACTIVE_DATASET", "nf_ton_iot_v2")

_active_profile = DATASET_PROFILES[ACTIVE_DATASET]
print(f"Active dataset profile: {ACTIVE_DATASET!r} — {_active_profile['description']}")

DEFAULT_DATASET_PATH = _active_profile["path"]
TARGET_COL = _active_profile["target_col"]
DROP_COLS_IF_PRESENT = _active_profile["drop_cols"]
IP_COLS_IF_PRESENT = _active_profile["ip_cols"]
ATTACK_COL = _active_profile.get("attack_col")                 # only set for *_binary profiles (evaluation-only column)
BINARY_LABEL_NAMES = _active_profile.get("binary_label_names")   # ("Benign", "Malicious") for *_binary profiles

# ESP32 constraints — these are host-side proxy estimates, NOT true ESP32
# measurements.  Always benchmark the generated embedded C on real hardware.
MAX_FLASH_KB = 320.0
MAX_LATENCY_MS = 0.10
MAX_COMPLEXITY = 2500.0
EMBEDDED_BYTES_PER_NODE_EST = 24

N_FOLDS_CV = 3
SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20

CLIP_MIN = -1e10
CLIP_MAX = 1e10

# Fitness weights.
FITNESS_WEIGHTS: Dict[str, float] = {
    "accuracy": 0.6,
    "macro_f1": 0.8,
    "mcc": 1.2,           # high weight: best single metric for imbalanced data
    "kappa": 1.0,
    "small_class_f1": 1.5,  # highest weight: rare-class F1
    "minority_recall": 1.4,
    "g_mean": 0.8,
    "weighted_f1": 0.3,
    # size penalties
    "size": 0.02,
    "latency": 0.01,
    "depth": 0.01,
    "leaves": 0.01,
    "constraint_penalty": 0.03,
}

DT_SEARCH_SPACE: Dict[str, List[Any]] = {
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_depth": [None] + list(range(3, 50, 2)),
    "min_samples_split": [2, 5, 10, 15, 20, 30, 40, 50],
    "min_samples_leaf": [1, 2, 4, 8, 12, 16, 20, 25],
    "min_weight_fraction_leaf": [0.0, 0.01, 0.02, 0.05, 0.08],
    "max_features": [None, "sqrt", "log2", 0.3, 0.5, 0.7],
    "class_weight": [None, "balanced"],
    "ccp_alpha": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
}

PRUNE_SEARCH_SPACE: Dict[str, List[Any]] = {
    "max_depth": [5, 6, 7, 8, 10, 12, 14, 16, 18, 20, None],
    "min_samples_split": [2, 5, 10, 15, 20, 30, 40],
    "min_samples_leaf": [1, 2, 4, 8, 12, 16, 20],
    "ccp_alpha": [1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2],
}

# -----------------------------------------------------------------------
# Shared output folders. Every figure and every ESP32 export is written into
# one of these two folders.
# -----------------------------------------------------------------------
# Namespaced by ACTIVE_DATASET so results from different dataset profiles
# never overwrite each other and can be compared side by side.
FIGURES_DIR = Path("figures") / ACTIVE_DATASET
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ESP32_DIR = Path("esp32_export") / ACTIVE_DATASET
ESP32_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed: int) -> None:
    """Set numpy and random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def sanitize_array(X: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Replace inf/nan with 0 and clip to safe range."""
    X = np.asarray(X, dtype=dtype)
    X = np.where(np.isfinite(X), X, 0.0)
    X = np.clip(X, CLIP_MIN, CLIP_MAX)
    return X.astype(dtype, copy=False)


class SafeFiniteTransformer(BaseEstimator, TransformerMixin):
    """Replace unsafe numeric values inside a sklearn Pipeline.

    SimpleImputer handles NaN but not +/-inf. DecisionTreeClassifier also
    validates inputs as float32 internally, so extremely large finite values can
    fail at predict time with:

        ValueError: Input X contains infinity or a value too large for dtype('float32')

    This transformer is intentionally used before the imputer and again after
    scaling/feature selection.  When ``inf_to_nan=True``, infinities become NaN
    so the imputer can fill them from training-fold medians.  Otherwise all
    non-finite values are replaced and everything is clipped to a safe range.
    """

    def __init__(
        self,
        nan_to: float = 0.0,
        clip_min: float = CLIP_MIN,
        clip_max: float = CLIP_MAX,
        inf_to_nan: bool = False,
    ) -> None:
        self.nan_to = nan_to
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.inf_to_nan = inf_to_nan

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=np.float64)
        if self.inf_to_nan:
            arr = arr.copy()
            arr[~np.isfinite(arr)] = np.nan
        else:
            arr = np.nan_to_num(
                arr,
                nan=self.nan_to,
                posinf=self.clip_max,
                neginf=self.clip_min,
            )
        arr = np.clip(arr, self.clip_min, self.clip_max)
        return arr.astype(np.float32, copy=False)


def _get_tree(model_or_pipeline) -> DecisionTreeClassifier:
    """Extract the DecisionTreeClassifier from a Pipeline or return as-is."""
    if isinstance(model_or_pipeline, Pipeline):
        return model_or_pipeline.named_steps["clf"]
    return model_or_pipeline


def count_nodes(model_or_pipeline) -> int:
    return int(_get_tree(model_or_pipeline).tree_.node_count)


def count_leaves(model_or_pipeline) -> int:
    return int(_get_tree(model_or_pipeline).get_n_leaves())


def count_depth(model_or_pipeline) -> int:
    return int(_get_tree(model_or_pipeline).get_depth())


def estimate_embedded_size_kb(model_or_pipeline) -> float:
    return float(count_nodes(model_or_pipeline) * EMBEDDED_BYTES_PER_NODE_EST) / 1024.0


def benchmark_latency(model, X_sample, n_runs: int = 5) -> float:
    """Estimate per-sample prediction latency in milliseconds.

    converts DataFrame slices to numpy so predict timing is not
    dominated by pandas overhead.
    """
    if len(X_sample) == 0:
        return 0.0
    # Take at most 100 rows; convert to numpy for clean timing
    sample = X_sample.iloc[:100].to_numpy() if hasattr(X_sample, "iloc") else X_sample[:100]
    try:
        for _ in range(2):          # warm-up
            model.predict(sample)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.predict(sample)
            times.append((time.perf_counter() - t0) / len(sample) * 1000)
        return float(np.median(times))
    except Exception:
        return 0.1


def compute_constraint_penalty(size_kb: float, latency_ms: float, leaf_count: int) -> float:
    """Return a combined penalty for violating ESP32 hardware constraints."""
    flash_over = max(0.0, size_kb - MAX_FLASH_KB) / MAX_FLASH_KB
    latency_over = max(0.0, latency_ms - MAX_LATENCY_MS) / MAX_LATENCY_MS
    complexity_over = max(0.0, leaf_count - MAX_COMPLEXITY) / MAX_COMPLEXITY
    return flash_over + latency_over + complexity_over


def infer_rare_classes(
    y_train: np.ndarray,
    rare_fraction: float = 0.25,
    extreme_fraction: float = 0.10,
) -> Tuple[Set[int], Set[int]]:
    """Infer rare / extremely-rare label sets from training frequencies only.

    Returns two sets of integer label indices sorted by ascending frequency.
    """
    labels, counts = np.unique(y_train, return_counts=True)
    order = labels[np.argsort(counts)]   # rarest first
    n = len(order)
    n_rare = max(1, math.ceil(n * rare_fraction))
    n_extreme = max(1, math.ceil(n * extreme_fraction))
    return set(order[:n_rare].tolist()), set(order[:n_extreme].tolist())


def _json_default(obj):
    """JSON serializer for numpy scalars and other non-serialisable types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# =============================================================================
# METRICS
# =============================================================================

def compute_enhanced_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rare_classes: Optional[Set[int]] = None,
    extremely_rare_classes: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics with MCC / Kappa emphasis.

    All label sets are derived from the union of y_true and y_pred so the
    function works correctly even if some classes are absent from one array.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    class_counts = np.array([(y_true == lab).sum() for lab in labels], dtype=float)

    recalls, precisions, f1s = [], [], []
    for i in range(len(labels)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    inv_weights = 1.0 / (class_counts + 1e-6)
    inv_weights = inv_weights / inv_weights.sum()
    weighted_small_f1 = float(np.dot(np.array(f1s), inv_weights))
    minority_recall = float(np.min(recalls)) if recalls else 0.0
    g_mean = float(np.exp(np.mean(np.log(np.array(recalls) + 1e-10)))) if recalls else 0.0

    overall_mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    per_class_mcc: Dict[int, float] = {}
    for lab in labels:
        per_class_mcc[int(lab)] = float(
            matthews_corrcoef((y_true == lab).astype(int), (y_pred == lab).astype(int))
        )

    rare_classes = set() if rare_classes is None else set(rare_classes)
    extremely_rare_classes = set() if extremely_rare_classes is None else set(extremely_rare_classes)
    rare_values = [per_class_mcc[int(lab)] for lab in rare_classes if int(lab) in per_class_mcc]
    extreme_values = [per_class_mcc[int(lab)] for lab in extremely_rare_classes if int(lab) in per_class_mcc]
    rare_mcc = float(np.mean(rare_values)) if rare_values else 0.0
    extremely_rare_mcc = float(np.mean(extreme_values)) if extreme_values else 0.0
    weighted_mcc = float(sum(per_class_mcc[int(lab)] * inv_weights[i] for i, lab in enumerate(labels)))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "small_class_f1": weighted_small_f1,
        "minority_recall": minority_recall,
        "g_mean": g_mean,
        "mcc": overall_mcc,
        "kappa": kappa,
        "per_class_recall": recalls,
        "per_class_precision": precisions,
        "per_class_f1": f1s,
        "per_class_mcc": per_class_mcc,
        "rare_class_mcc": rare_mcc,
        "extremely_rare_mcc": extremely_rare_mcc,
        "weighted_mcc": weighted_mcc,
    }


# =============================================================================
# PIPELINE BUILDER + CV EVALUATION  (no preprocessing leakage)
# =============================================================================

def build_pipeline(params: Dict[str, Any], k: int, seed: int) -> Pipeline:
    """Build a full safe sklearn Pipeline.

    Keeping preprocessing inside the Pipeline ensures that each CV fold fits
    the imputer, scaler, and feature selector only on its own training portion.
    SafeFiniteTransformer prevents +/-inf and too-large float32 values
    from reaching DecisionTreeClassifier during fit or predict.
    """
    return Pipeline([
        # Convert +/-inf to NaN so the imputer learns medians from the training fold only.
        ("finite_before_imputer", SafeFiniteTransformer(inf_to_nan=True)),
        ("imputer",  SimpleImputer(strategy="median")),
        ("finite_after_imputer", SafeFiniteTransformer()),
        ("scaler",   RobustScaler(quantile_range=(1, 99))),
        ("finite_after_scaler", SafeFiniteTransformer()),
        ("selector", SelectKBest(score_func=f_classif, k=k)),
        ("finite_after_selector", SafeFiniteTransformer()),
        ("clf",      DecisionTreeClassifier(random_state=seed, **params)),
    ])


def evaluate_pipeline_cv(
    params: Dict[str, Any],
    k: int,
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    rare_classes: Set[int],
    extremely_rare_classes: Set[int],
) -> Dict[str, Any]:
    """Cross-validate a (params, k) combination and return aggregated metrics.

    Returns a dict that includes a 'fitted_pipeline' key containing the model
    trained on the full X/y (used for size/latency estimates and later for
    saving).
    """
    k_eff = min(k, X.shape[1])
    skf = StratifiedKFold(n_splits=N_FOLDS_CV, shuffle=True, random_state=seed)

    fold_metrics: List[Dict[str, Any]] = []
    all_y_true: List[int] = []
    all_y_pred: List[int] = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        pipe = build_pipeline(params, k_eff, seed)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_val)
        fold_metrics.append(compute_enhanced_metrics(y_val, y_pred, rare_classes, extremely_rare_classes))
        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred.tolist())

    # Fit once on all provided X for size/latency estimates
    fitted_all = build_pipeline(params, k_eff, seed)
    fitted_all.fit(X, y)

    def _mean(name: str) -> float:
        return float(np.mean([m[name] for m in fold_metrics]))

    def _std(name: str) -> float:
        return float(np.std([m[name] for m in fold_metrics]))

    final_metrics = compute_enhanced_metrics(
        np.array(all_y_true), np.array(all_y_pred), rare_classes, extremely_rare_classes
    )

    nodes       = count_nodes(fitted_all)
    leaves      = count_leaves(fitted_all)
    depth       = count_depth(fitted_all)
    embedded_kb = estimate_embedded_size_kb(fitted_all)
    latency     = benchmark_latency(fitted_all, X)
    penalty     = compute_constraint_penalty(embedded_kb, latency, leaves)

    return {
        "cv_accuracy_mean":           _mean("accuracy"),
        "cv_macro_f1_mean":           _mean("macro_f1"),
        "cv_weighted_f1_mean":        _mean("weighted_f1"),
        "cv_small_class_f1_mean":     _mean("small_class_f1"),
        "cv_minority_recall_mean":    _mean("minority_recall"),
        "cv_g_mean_mean":             _mean("g_mean"),
        "cv_mcc_mean":                _mean("mcc"),
        "cv_kappa_mean":              _mean("kappa"),
        "cv_rare_class_mcc_mean":     _mean("rare_class_mcc"),
        "cv_extremely_rare_mcc_mean": _mean("extremely_rare_mcc"),
        "cv_mcc_std":                 _std("mcc"),
        "cv_kappa_std":               _std("kappa"),
        "final_mcc":                  final_metrics["mcc"],
        "final_kappa":                final_metrics["kappa"],
        "final_rare_class_mcc":       final_metrics["rare_class_mcc"],
        "final_small_class_f1":       final_metrics["small_class_f1"],
        "node_count":                 nodes,
        "leaf_count":                 leaves,
        "depth":                      depth,
        "embedded_size_kb":           embedded_kb,
        "latency_ms_per_sample":      latency,
        "constraint_penalty":         penalty,
        "meets_constraints":          (embedded_kb <= MAX_FLASH_KB
                                       and latency <= MAX_LATENCY_MS
                                       and leaves <= MAX_COMPLEXITY),
        "fitted_pipeline":            fitted_all,   # popped in evaluate_candidate
    }


def compute_fitness(cv_results: Dict[str, Any]) -> float:
    """Compute a scalar fitness score from cross-validation results.

    Higher is better.  Bonus for rare-class MCC is added separately so it
    can be tracked independently.
    """
    w = FITNESS_WEIGHTS
    fitness = (
        w["accuracy"]           * cv_results["cv_accuracy_mean"]
        + w["macro_f1"]         * cv_results["cv_macro_f1_mean"]
        + w["mcc"]              * cv_results["cv_mcc_mean"]
        + w["kappa"]            * cv_results["cv_kappa_mean"]
        + w["small_class_f1"]   * cv_results["cv_small_class_f1_mean"]
        + w["minority_recall"]  * cv_results["cv_minority_recall_mean"]
        + w["g_mean"]           * cv_results["cv_g_mean_mean"]
        + w["weighted_f1"]      * cv_results["cv_weighted_f1_mean"]
        - w["size"]             * (cv_results["embedded_size_kb"] / MAX_FLASH_KB)
        - w["latency"]          * (cv_results["latency_ms_per_sample"] / MAX_LATENCY_MS)
        - w["depth"]            * (cv_results["depth"] / 50)
        - w["leaves"]           * (cv_results["leaf_count"] / MAX_COMPLEXITY)
        - w["constraint_penalty"] * cv_results["constraint_penalty"]
    )
    fitness += 0.5 * cv_results.get("cv_rare_class_mcc_mean", 0.0)
    return float(fitness)


def evaluate_candidate(
    params: Dict[str, Any],
    k: int,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
    rare_classes: Set[int],
    extremely_rare_classes: Set[int],
) -> Dict[str, Any]:
    """Evaluate a single (params, k) candidate; return a lightweight dict.

    only params and scalar metrics are stored in the returned dict.
    The fitted_pipeline is popped out and kept separately to avoid deepcopy of
    heavy sklearn objects during population evolution.
    """
    metrics = evaluate_pipeline_cv(params, k, X_train, y_train, seed, rare_classes, extremely_rare_classes)
    fitted_pipeline = metrics.pop("fitted_pipeline")
    metrics["fitness"] = compute_fitness(metrics)
    metrics["params"] = copy.deepcopy(params)
    metrics["k"] = k
    return {
        "params":   copy.deepcopy(params),
        "pipeline": fitted_pipeline,          # not deepcopied in population
        "metrics":  metrics,
    }


# =============================================================================
# SILVERBACK — global evolutionary search with memory
# =============================================================================

class EnhancedSilverback:
    """Global evolutionary search with adaptive mutation, crossover, and novelty memory."""

    def __init__(self, param_space: Dict[str, List[Any]], seed: int = 42):
        self.param_space = param_space
        self.rng = random.Random(seed)
        self.memory: deque = deque(maxlen=15)

    def sample_params(self) -> Dict[str, Any]:
        return {k: self.rng.choice(v) for k, v in self.param_space.items()}

    def adaptive_mutate(self, params: Dict[str, Any], iteration: int, total: int) -> Dict[str, Any]:
        """Mutate with strength that decays over time (high early, low late)."""
        strength = 0.4 * (1.0 - iteration / max(total, 1)) + 0.1
        if self.rng.random() < 0.15:
            strength *= 1.5
        new_params = copy.deepcopy(params)
        keys = list(self.param_space.keys())
        n_mutate = max(1, min(len(keys), int(len(keys) * strength)))
        for key in self.rng.sample(keys, n_mutate):
            available = [v for v in self.param_space[key] if v != new_params.get(key)]
            if available:
                new_params[key] = self.rng.choice(available)
        return new_params

    def crossover(self, p1: Dict[str, Any], p2: Dict[str, Any]) -> Dict[str, Any]:
        return {k: (p1[k] if self.rng.random() < 0.5 else p2[k]) for k in self.param_space}

    def _normalize(self, key: str, value: Any) -> float:
        vals = self.param_space[key]
        return vals.index(value) / max(len(vals) - 1, 1) if value in vals else 0.5

    def param_distance(self, p1: Dict[str, Any], p2: Dict[str, Any]) -> float:
        return float(np.sqrt(sum(
            (self._normalize(k, p1.get(k)) - self._normalize(k, p2.get(k))) ** 2
            for k in self.param_space
        )))

    def novelty_bonus(self, params: Dict[str, Any]) -> float:
        if not self.memory:
            return 0.05
        min_dist = min(self.param_distance(params, m) for m in self.memory)
        return float(0.08 * (min_dist / 0.2)) if min_dist > 0.2 else 0.0


def run_silverback_enhanced(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    k: int,
    budget: int,
    seed: int,
    rare_classes: Set[int],
    extremely_rare_classes: Set[int],
    verbose: bool = True,
) -> Tuple[Dict, List]:
    """Run Silverback global search; returns (best_candidate, history)."""
    silverback = EnhancedSilverback(DT_SEARCH_SPACE, seed)
    pop_size = min(10, max(6, budget // 5))
    population: List[Dict] = []
    history: List[Dict] = []
    evals = 0

    if verbose:
        print(f"    Initialising population ({pop_size} individuals)…")
    while len(population) < pop_size and evals < budget:
        params = silverback.sample_params()
        cand = evaluate_candidate(params, k, X_train, y_train, seed, rare_classes, extremely_rare_classes)
        population.append(cand)
        silverback.memory.append(params)
        history.append(cand["metrics"])
        evals += 1

    best_fitness = max(c["metrics"]["fitness"] for c in population)
    stagnation = 0
    iteration = 0

    while evals < budget:
        population.sort(key=lambda x: x["metrics"]["fitness"], reverse=True)
        elite_count = max(2, len(population) // 3)
        # guard so we never try to index elites[1] when there is only 1
        elites = population[:elite_count]
        safe_elites = elites if len(elites) >= 2 else elites * 2

        current_best = population[0]["metrics"]["fitness"]
        if current_best > best_fitness + 1e-6:
            best_fitness = current_best
            stagnation = 0
        else:
            stagnation += 1

        # new_population keeps only params+metrics, not deep-copied pipelines
        new_population = [
            {"params": copy.deepcopy(safe_elites[0]["params"]),
             "pipeline": safe_elites[0]["pipeline"],
             "metrics": copy.deepcopy(safe_elites[0]["metrics"])},
            {"params": copy.deepcopy(safe_elites[1]["params"]),
             "pipeline": safe_elites[1]["pipeline"],
             "metrics": copy.deepcopy(safe_elites[1]["metrics"])},
        ]

        while len(new_population) < pop_size and evals < budget:
            if silverback.rng.random() < 0.6:
                p1, p2 = silverback.rng.sample(safe_elites, 2)
                child_params = silverback.crossover(p1["params"], p2["params"])
            else:
                parent = silverback.rng.choice(safe_elites)
                child_params = silverback.adaptive_mutate(parent["params"], iteration, budget)

            cand = evaluate_candidate(child_params, k, X_train, y_train, seed, rare_classes, extremely_rare_classes)
            cand["metrics"]["fitness"] += silverback.novelty_bonus(child_params)
            new_population.append(cand)
            silverback.memory.append(child_params)
            history.append(cand["metrics"])
            evals += 1

        population = new_population

        if stagnation > 8 and evals < budget:
            if verbose:
                print(f"    Diversity injection (stagnation={stagnation})…")
            params = silverback.sample_params()
            cand = evaluate_candidate(params, k, X_train, y_train, seed, rare_classes, extremely_rare_classes)
            population.append(cand)
            history.append(cand["metrics"])
            evals += 1
            stagnation = 0

        iteration += 1

    best = max(population, key=lambda x: x["metrics"]["fitness"])
    return best, history


# =============================================================================
# DOLPHIN — local search with simulated annealing
# =============================================================================

class EnhancedDolphin:
    """Local search with simulated annealing, adaptive mutation, and elite memory."""

    def __init__(self, param_space: Dict[str, List[Any]], seed: int = 42):
        self.param_space = param_space
        self.rng = random.Random(seed + 999)
        self.elite_memory: deque = deque(maxlen=8)
        self.mutation_strength = 0.25
        self.temperature = 1.0

    def local_mutate(self, params: Dict[str, Any], strength: float) -> Dict[str, Any]:
        new_params = copy.deepcopy(params)
        keys = list(self.param_space.keys())
        n_mutate = max(1, min(len(keys), int(len(keys) * strength)))
        for key in self.rng.sample(keys, n_mutate):
            available = [v for v in self.param_space[key] if v != new_params.get(key)]
            if available and self.rng.random() < 0.7:
                new_params[key] = self.rng.choice(available)
        return new_params

    def crossover_with_elite(self, current: Dict[str, Any], elite: Dict[str, Any], rate: float = 0.3) -> Dict[str, Any]:
        child = copy.deepcopy(current)
        for key in self.param_space:
            if self.rng.random() < rate:
                child[key] = elite[key]
        return child


def run_dolphin_enhanced(
    start_params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    k: int,
    budget: int,
    seed: int,
    rare_classes: Set[int],
    extremely_rare_classes: Set[int],
    verbose: bool = True,
) -> Tuple[Dict, List]:
    """Run Dolphin local refinement from start_params; returns (best_candidate, history)."""
    dolphin = EnhancedDolphin(DT_SEARCH_SPACE, seed)
    current = evaluate_candidate(start_params, k, X_train, y_train, seed, rare_classes, extremely_rare_classes)
    best = {"params": copy.deepcopy(current["params"]),
            "pipeline": current["pipeline"],
            "metrics": copy.deepcopy(current["metrics"])}
    history = [current["metrics"]]
    dolphin.elite_memory.append(best)

    if verbose:
        print(f"    Initial MCC:   {current['metrics']['cv_mcc_mean']:.4f}")
        print(f"    Initial Kappa: {current['metrics']['cv_kappa_mean']:.4f}")

    no_improve = 0
    evals = 1

    for step in range(budget - 1):
        dolphin.temperature = max(0.05, 1.0 - (step / max(budget, 1)) ** 1.5)
        strength = max(0.08, dolphin.mutation_strength * dolphin.temperature)

        if len(dolphin.elite_memory) > 1 and dolphin.rng.random() < 0.3:
            elite = dolphin.rng.choice(list(dolphin.elite_memory))
            child_params = dolphin.crossover_with_elite(current["params"], elite["params"])
        else:
            child_params = dolphin.local_mutate(current["params"], strength)

        child = evaluate_candidate(child_params, k, X_train, y_train, seed, rare_classes, extremely_rare_classes)
        history.append(child["metrics"])
        evals += 1

        delta = child["metrics"]["fitness"] - current["metrics"]["fitness"]
        accept = delta > 0 or dolphin.rng.random() < math.exp(delta / (dolphin.temperature + 1e-9))

        if accept:
            current = child
            if child["metrics"]["fitness"] > best["metrics"]["fitness"]:
                best = {"params": copy.deepcopy(child["params"]),
                        "pipeline": child["pipeline"],
                        "metrics": copy.deepcopy(child["metrics"])}
                dolphin.elite_memory.append(best)
                dolphin.mutation_strength = min(0.35, dolphin.mutation_strength * 1.05)
                no_improve = 0
                if verbose and step % 5 == 0:
                    print(f"      Step {step:3d}: new best MCC={best['metrics']['cv_mcc_mean']:.4f}")
            else:
                no_improve += 1
        else:
            dolphin.mutation_strength = max(0.08, dolphin.mutation_strength * 0.95)
            no_improve += 1

        if no_improve >= 10 and dolphin.elite_memory:
            if verbose:
                print(f"      Reset at step {step} (stagnation)")
            current = copy.deepcopy(dolphin.rng.choice(list(dolphin.elite_memory)))
            dolphin.mutation_strength = 0.30
            no_improve = 0

    if verbose:
        print(f"    Final MCC:   {best['metrics']['cv_mcc_mean']:.4f}")
        print(f"    Final Kappa: {best['metrics']['cv_kappa_mean']:.4f}")
    return best, history


# =============================================================================
# PRUNING SILVERBACK — stage-3 fine-tuning of pruning hyperparameters
# =============================================================================

def run_prune_silverback(
    base_params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    k: int,
    budget: int,
    seed: int,
    rare_classes: Set[int],
    extremely_rare_classes: Set[int],
) -> Tuple[Dict, List]:
    """Silverback search over pruning parameters only; returns (best_candidate, history)."""
    silverback = EnhancedSilverback(PRUNE_SEARCH_SPACE, seed + 1000)
    pop_size = min(8, max(5, budget // 4))
    population: List[Dict] = []
    history: List[Dict] = []
    evals = 0

    def apply_pruning(prune_params: Dict[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(base_params)
        merged.update(prune_params)
        return merged

    while len(population) < pop_size and evals < budget:
        prune_params = silverback.sample_params()
        params = apply_pruning(prune_params)
        cand = evaluate_candidate(params, k, X_train, y_train, seed, rare_classes, extremely_rare_classes)
        cand["prune_params"] = prune_params
        population.append(cand)
        history.append(cand["metrics"])
        evals += 1

    iteration = 0
    while evals < budget:
        population.sort(key=lambda x: x["metrics"]["fitness"], reverse=True)
        elite_count = max(2, len(population) // 3)
        elites = population[:elite_count]
        # guard
        safe_elites = elites if len(elites) >= 2 else elites * 2

        new_population = [
            {"params": copy.deepcopy(safe_elites[0]["params"]),
             "pipeline": safe_elites[0]["pipeline"],
             "metrics": copy.deepcopy(safe_elites[0]["metrics"]),
             "prune_params": copy.deepcopy(safe_elites[0]["prune_params"])},
            {"params": copy.deepcopy(safe_elites[1]["params"]),
             "pipeline": safe_elites[1]["pipeline"],
             "metrics": copy.deepcopy(safe_elites[1]["metrics"]),
             "prune_params": copy.deepcopy(safe_elites[1]["prune_params"])},
        ]

        while len(new_population) < pop_size and evals < budget:
            parent = silverback.rng.choice(safe_elites)
            child_prune = silverback.adaptive_mutate(parent["prune_params"], iteration, budget)
            params = apply_pruning(child_prune)
            cand = evaluate_candidate(params, k, X_train, y_train, seed, rare_classes, extremely_rare_classes)
            cand["prune_params"] = child_prune
            new_population.append(cand)
            history.append(cand["metrics"])
            evals += 1

        population = new_population
        iteration += 1

    best = max(population, key=lambda x: x["metrics"]["fitness"])
    return best, history



# =============================================================================
# DATA HELPERS
# =============================================================================

# Identifier/leaky column *patterns* (matched case/underscore/space-insensitively
# as substrings), used as a safety net on top of each profile's explicit
# drop_cols/ip_cols list. Different CICFlowMeter/NetFlow exports name these
# columns very differently -- "Src IP" vs "src_ip" vs "IPV4_SRC_ADDR" vs
# "srcip" -- so an exact-string list alone is brittle across dataset
# profiles. This keeps the framework leakage-safe even when a new profile's
# exact column names weren't known in advance. It never touches TARGET_COL.
LEAKY_COLUMN_PATTERNS = ["ip", "port", "timestamp", "flowid", "mac"]


def _normalize_col(name: str) -> str:
    return str(name).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def load_and_prepare_dataframe(data_path: str) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder, List[str], tuple]:
    """Load CSV, drop unwanted/leaky columns, encode target, return numeric-only features.

    Memory-efficient version: only reads the columns that will actually be
    kept (via `usecols`), and loads non-target columns as float32 instead of
    pandas' default float64/int64. For a wide, many-row CSV this roughly
    halves peak memory and avoids the single-block consolidation blowing up
    (e.g. MemoryError on very large row counts).
    """
    # 0) Peek at the header only (no row data) so column decisions are made
    #    before paying the cost of loading any rows.
    header_cols = pd.read_csv(data_path, nrows=0).columns.tolist()

    # 1) Explicit drops from the active profile (exact names).
    explicit_drop = [c for c in DROP_COLS_IF_PRESENT + IP_COLS_IF_PRESENT if c in header_cols]

    # 2) Pattern-based safety net for identifier/leaky columns whose exact
    #    name wasn't anticipated by the profile (see LEAKY_COLUMN_PATTERNS).
    pattern_drop = [
        c for c in header_cols
        if _normalize_col(c) != _normalize_col(TARGET_COL)
        and c not in explicit_drop
        and any(p in _normalize_col(c) for p in LEAKY_COLUMN_PATTERNS)
    ]
    if explicit_drop:
        print(f"  Dropping column(s) (explicit profile list): {explicit_drop}")
    if pattern_drop:
        print(f"  Dropping column(s) (pattern-matched as identifier/leaky): {pattern_drop}")

    usecols = [c for c in header_cols if c not in explicit_drop and c not in pattern_drop]

    # 3) Resolve the target column, falling back to a case/format-insensitive
    #    match if the exact TARGET_COL string isn't present (e.g. profile
    #    says "Attack" but the file has "attack").
    target_col = TARGET_COL
    if target_col not in usecols:
        candidates = [c for c in usecols if _normalize_col(c) == _normalize_col(TARGET_COL)]
        if candidates:
            target_col = candidates[0]
            print(f"  Note: target_col {TARGET_COL!r} matched case/format-insensitively to {target_col!r}")
        else:
            raise ValueError(
                f"Target column {TARGET_COL!r} not found. "
                f"Available columns: {usecols[:20]} …"
            )

    # 4) Load only the surviving columns, downcasting everything except the
    #    target to float32 (roughly half the memory of pandas' float64/int64
    #    defaults). If any "numeric-looking" feature column actually contains
    #    stray non-numeric tokens, fall back to a plain read of just the
    #    needed columns -- still far cheaper than loading the full file.
    dtype_map = {c: "float32" for c in usecols if c != target_col}
    try:
        df = pd.read_csv(data_path, usecols=usecols, dtype=dtype_map)
    except (ValueError, TypeError):
        df = pd.read_csv(data_path, usecols=usecols)

    if BINARY_LABEL_NAMES:
        # binary task: the 0/1 Label column -> y in {0, 1}; classes named so that 0 = Benign, 1 = Malicious (the positive class)
        raw = pd.to_numeric(df[target_col], errors="coerce")
        if raw.isna().any() or not set(np.unique(raw.dropna().astype(int)).tolist()) <= {0, 1}:
            raise ValueError(f"binary target {target_col!r} must contain only 0/1 values")
        y = raw.to_numpy().astype(np.int64)
        le = LabelEncoder(); le.classes_ = np.array(list(BINARY_LABEL_NAMES))
    else:
        le = LabelEncoder()
        y = le.fit_transform(df[target_col])
    X_all = df.drop(columns=[target_col])
    numeric_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric feature columns found after dropping target/IP/label columns.")
    X = X_all[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return X, y, le, numeric_cols, df.shape


def stratified_sample(
    X: pd.DataFrame, y: np.ndarray, sample_limit: int, seed: int
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Return a stratified sample of at most `sample_limit` rows."""
    if len(X) <= sample_limit:
        return X, y
    frac = sample_limit / len(X)
    X_s, _, y_s, _ = train_test_split(X, y, train_size=frac, stratify=y, random_state=seed)
    return X_s, y_s


def evaluate_on_split(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    rare_classes: Set[int],
    extremely_rare_classes: Set[int],
) -> Dict[str, Any]:
    """Evaluate a fitted pipeline on a held-out split and append size metrics."""
    y_pred = pipe.predict(X)
    metrics = compute_enhanced_metrics(y, y_pred, rare_classes, extremely_rare_classes)
    metrics.update({
        "node_count":            count_nodes(pipe),
        "leaf_count":            count_leaves(pipe),
        "depth":                 count_depth(pipe),
        "embedded_size_kb":      estimate_embedded_size_kb(pipe),
        "latency_ms_per_sample": benchmark_latency(pipe, X),
    })
    return metrics



# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main(overrides=None):
    """End-to-end optimisation pipeline.

    Flow:
      1. Load & split data (train / val / test).  Test is never touched until
         the very end.
      2. For each feature count K: run Silverback → Dolphin → Pruning search,
         each using cross-validation with the full preprocessing Pipeline
         inside the fold.
      3. Select the best (K, params) combination by *validation* MCC.
      4. Retrain the winner on train+val.
      5. Evaluate on test exactly once and save the full artifact.
    """
    set_seed((overrides or {}).get('seed', 42))  # reproducible even when called as a module

    print("=" * 70)
    print("FaithTree decision-tree search")
    print("No test-set selection | Safe Pipeline CV | Full artifact saved")
    print("=" * 70)

    config: Dict[str, Any] = {
        "data_path":          DEFAULT_DATASET_PATH,
        "target_col":         TARGET_COL,
        "test_size":          0.20,
        "val_size":           0.20,
        "seed":               42,
        "ks":                 [20, 25, 30, 35, 40],
        "silverback_budget":  45,
        "dolphin_budget":     30,
        "prune_budget":       20,
        "sample_limit":       300_000,  # large enough that ultra-rare classes (dos/ddos/mitm in CIC-ToN-IoT) are represented during the search
    }
    if overrides:                      # optional: quick smoke-runs / budget studies
        config.update(overrides)
    print("\nConfiguration:")
    for key, val in config.items():
        print(f"  {key}: {val}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n📂 Loading dataset…")
    X, y, le, numeric_cols, original_shape = load_and_prepare_dataframe(config["data_path"])
    print(f"  Original shape : {original_shape}")
    print(f"  Numeric features: {len(numeric_cols)}")
    print(f"  Target classes  : {len(le.classes_)}")
    for i, cnt in enumerate(np.bincount(y)):
        print(f"    [{i}] {le.classes_[i]:30s} {cnt:>8,}  ({cnt / len(y) * 100:.2f}%)")

    # ------------------------------------------------------------------
    # Split  (test is truly held-out)
    # ------------------------------------------------------------------
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=config["test_size"], stratify=y, random_state=config["seed"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=config["val_size"], stratify=y_train_val, random_state=config["seed"]
    )

    # rare classes from training labels only
    rare_classes, extremely_rare_classes = infer_rare_classes(y_train)
    print(f"\nRare classes (inferred):          {sorted(rare_classes)}")
    print(f"Extremely rare classes (inferred): {sorted(extremely_rare_classes)}")

    print("\n📊 Data split:")
    print(f"  Train      : {len(X_train):>8,}")
    print(f"  Validation : {len(X_val):>8,}")
    print(f"  Test       : {len(X_test):>8,}  ← held-out until final reporting")

    # Stratified sample for the search phase
    X_tr_sample, y_tr_sample = stratified_sample(
        X_train, y_train, config["sample_limit"], config["seed"]
    )
    if len(X_tr_sample) < len(X_train):
        print(f"\n  (Search uses stratified sample: {len(X_tr_sample):,} rows)")

    # ------------------------------------------------------------------
    # Search loop over K values
    # ------------------------------------------------------------------
    validation_results: List[Dict[str, Any]] = []
    search_history_rows: List[Dict[str, Any]] = []  # for convergence figure

    for k_idx, k in enumerate(config["ks"]):
        k_eff = min(k, len(numeric_cols))
        print(f"\n{'=' * 60}")
        print(f"🔍 K={k_eff}  (subset {k_idx + 1}/{len(config['ks'])})")
        print(f"{'=' * 60}")

        seed_k = config["seed"] + k_eff

        # Stage 1: Silverback
        print("\n  🦍 Stage 1: Silverback global search…")
        silver_best, silver_hist = run_silverback_enhanced(
            X_tr_sample, y_tr_sample, k_eff,
            config["silverback_budget"], seed_k,
            rare_classes, extremely_rare_classes, verbose=True,
        )
        print(f"    Silverback CV MCC:   {silver_best['metrics']['cv_mcc_mean']:.4f}")
        print(f"    Silverback CV Kappa: {silver_best['metrics']['cv_kappa_mean']:.4f}")

        # Stage 2: Dolphin
        print("\n  🐬 Stage 2: Dolphin local refinement…")
        dolphin_best, dolphin_hist = run_dolphin_enhanced(
            silver_best["params"], X_tr_sample, y_tr_sample, k_eff,
            config["dolphin_budget"], seed_k,
            rare_classes, extremely_rare_classes, verbose=True,
        )

        # Stage 3: Pruning
        print("\n  ✂️  Stage 3: Pruning optimisation…")
        prune_best, prune_hist = run_prune_silverback(
            dolphin_best["params"], X_tr_sample, y_tr_sample, k_eff,
            config["prune_budget"], seed_k,
            rare_classes, extremely_rare_classes,
        )

        for _stage, _hist in (("silverback", silver_hist), ("dolphin", dolphin_hist), ("prune", prune_hist)):
            for _i, _m in enumerate(_hist):
                search_history_rows.append({
                    "k": k_eff, "stage": _stage, "eval_in_stage": _i,
                    "fitness": _m.get("fitness"), "cv_mcc_mean": _m.get("cv_mcc_mean"),
                    "cv_macro_f1_mean": _m.get("cv_macro_f1_mean"),
                    "cv_small_class_f1_mean": _m.get("cv_small_class_f1_mean"),
                    "node_count": _m.get("node_count"), "embedded_size_kb": _m.get("embedded_size_kb"),
                })

        # ------------------------------------------------------------------
        # Validation scoring (used for model selection — NOT test)
        # ------------------------------------------------------------------
        val_pipe = build_pipeline(prune_best["params"], k_eff, seed_k)
        val_pipe.fit(X_train, y_train)
        val_metrics = evaluate_on_split(val_pipe, X_val, y_val, rare_classes, extremely_rare_classes)

        # get_feature_names_out() with no positional args
        try:
            selected_feats = list(val_pipe.named_steps["selector"].get_feature_names_out())
        except Exception:
            selected_feats = []

        print("\n  ✅ Validation results (used for model selection):")
        print(f"      Accuracy        : {val_metrics['accuracy']:.4f}")
        print(f"      Macro F1        : {val_metrics['macro_f1']:.4f}")
        print(f"      MCC             : {val_metrics['mcc']:.4f}  ★")
        print(f"      Kappa           : {val_metrics['kappa']:.4f}")
        print(f"      Small-class F1  : {val_metrics['small_class_f1']:.4f}")
        print(f"      Rare-class MCC  : {val_metrics['rare_class_mcc']:.4f}")
        print(f"      Size proxy      : {val_metrics['embedded_size_kb']:.2f} KB")

        validation_results.append({
            "k":                   k_eff,
            "params":              prune_best["params"],
            "validation_mcc":      val_metrics["mcc"],
            "validation_metrics":  val_metrics,
            "selected_features":   selected_feats,
        })
        gc.collect()

    # ------------------------------------------------------------------
    # Model selection: pick by validation MCC
    # ------------------------------------------------------------------
    best_val = max(validation_results, key=lambda r: r["validation_mcc"])
    print("\n" + "=" * 70)
    print("🏆 SELECTED MODEL (by validation MCC)")
    print("=" * 70)
    print(f"  K={best_val['k']},  validation MCC={best_val['validation_mcc']:.4f}")

    # ------------------------------------------------------------------
    # Final model: retrain on train + val, evaluate test ONCE
    # ------------------------------------------------------------------
    X_final = pd.concat([X_train, X_val], axis=0)
    y_final = np.concatenate([y_train, y_val])
    final_pipe = build_pipeline(best_val["params"], best_val["k"], config["seed"])
    final_pipe.fit(X_final, y_final)

    test_metrics = evaluate_on_split(final_pipe, X_test, y_test, rare_classes, extremely_rare_classes)

    print("\n🎯 Final test results (reported once):")
    print(f"  Accuracy        : {test_metrics['accuracy']:.4f}")
    print(f"  Macro F1        : {test_metrics['macro_f1']:.4f}")
    print(f"  MCC             : {test_metrics['mcc']:.4f}  ★")
    print(f"  Kappa           : {test_metrics['kappa']:.4f}")
    print(f"  Small-class F1  : {test_metrics['small_class_f1']:.4f}")
    print(f"  Minority recall : {test_metrics['minority_recall']:.4f}")
    print(f"  Rare-class MCC  : {test_metrics['rare_class_mcc']:.4f}")
    print(f"  Size proxy      : {test_metrics['embedded_size_kb']:.2f} KB")
    print(f"  Meets constraints: {test_metrics['meets_constraints'] if 'meets_constraints' in test_metrics else 'see constraint check'}")

    # ------------------------------------------------------------------
    # Save full pipeline artifact
    # ------------------------------------------------------------------
    output_dir = Path("esp32_dt_final_results") / ACTIVE_DATASET
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    artifact = {
        "pipeline":                final_pipe,
        "label_encoder":           le,
        "numeric_cols":            numeric_cols,
        "target_col":              TARGET_COL,
        "dropped_cols":            DROP_COLS_IF_PRESENT + IP_COLS_IF_PRESENT,
        "rare_classes":            sorted(rare_classes),
        "extremely_rare_classes":  sorted(extremely_rare_classes),
    }
    model_path = output_dir / f"full_pipeline_{stamp}.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(artifact, fh)

    # get_feature_names_out() — no positional args
    try:
        final_feats = list(final_pipe.named_steps["selector"].get_feature_names_out())
    except Exception:
        final_feats = []

    # Scalar-only sub-dicts for JSON
    def _scalar_dict(d: Dict) -> Dict:
        return {k: v for k, v in d.items() if isinstance(v, (int, float, bool, str, type(None)))}

    metadata = {
        "selection_rule":          "best validation MCC; test evaluated once after selection",
        "best_k":                  best_val["k"],
        "best_params":             best_val["params"],
        "validation_metrics":      _scalar_dict(best_val["validation_metrics"]),
        "test_metrics":            _scalar_dict(test_metrics),
        "numeric_cols":            numeric_cols,
        "selected_features_final": final_feats,
        "classes":                 le.classes_.tolist(),
        "rare_classes":            sorted(rare_classes),
        "extremely_rare_classes":  sorted(extremely_rare_classes),
        "model_path":              str(model_path),
        "note": (
            "ESP32 size/latency values are host-side proxy estimates. "
            "Always benchmark the generated embedded C on real hardware before deployment."
        ),
    }
    meta_path = output_dir / f"metadata_{stamp}.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=_json_default)
    # Summary CSV
    results_df = pd.DataFrame([
        {
            "k":                      r["k"],
            "validation_mcc":         r["validation_metrics"]["mcc"],
            "validation_accuracy":    r["validation_metrics"]["accuracy"],
            "validation_macro_f1":    r["validation_metrics"]["macro_f1"],
            "validation_small_cls_f1": r["validation_metrics"]["small_class_f1"],
            "model_size_kb_proxy":    r["validation_metrics"]["embedded_size_kb"],
        }
        for r in validation_results
    ])
    csv_path = output_dir / f"validation_summary_{stamp}.csv"
    results_df.to_csv(csv_path, index=False)
    pd.DataFrame(search_history_rows).to_csv(output_dir / f"search_history_{stamp}.csv", index=False)

    print(f"\n✅ Full pipeline artifact : {model_path}")
    print(f"✅ Metadata JSON          : {meta_path}")
    print(f"✅ Validation summary CSV : {csv_path}")

    return final_pipe, le, metadata, validation_results

# Imbalance handling: keep original distribution; compare unweighted vs class-weighted DT.
CLASS_WEIGHT_OPTIONS = [None, "balanced"]
