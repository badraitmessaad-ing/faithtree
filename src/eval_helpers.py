"""
eval_helpers.py -- evaluation helpers shared by notebooks 02, 03 and 04
(artifact loading, per-class metrics, same-split baselines, ablations, plots).
Import after changing into <repo>/results (the notebooks' first cell does this).
"""
import json, pickle, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             matthews_corrcoef, cohen_kappa_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

import framework as fw
from framework import *            # noqa: F401,F403  (config, SafeFiniteTransformer, build_pipeline, ...)
from framework import _json_default, _normalize_col   # noqa: F401  (underscore names are not star-imported)

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

DATASET_PATH = DEFAULT_DATASET_PATH
PUBLISHABILITY_OUTPUT_DIR = Path("publishability_checks") / ACTIVE_DATASET
PUBLISHABILITY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# FIX: the original notebook used FIGURES_DIR / ACTIVE_DATASET, which created figures/<ds>/<ds>/ .
PLOTS_DIR = FIGURES_DIR
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
ARTICLE_OUTPUT_DIR = Path("article_results") / ACTIVE_DATASET
ARTICLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_latest_artifact(dataset=None):
    """Newest full_pipeline_*.pkl (+ its metadata_*.json) written by notebook 01 for this dataset."""
    d = Path("esp32_dt_final_results") / (dataset or ACTIVE_DATASET)
    pkls = sorted(d.glob("full_pipeline_*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No trained model in {d.resolve()}. Run notebook 01 first.")
    pkl = pkls[-1]
    meta = d / pkl.name.replace("full_pipeline_", "metadata_").replace(".pkl", ".json")
    return pkl, (meta if meta.exists() else None)


def _expose_framework_classes():
    """Older pickles reference the pipeline classes as __main__.<name> (they were saved from a notebook).
    Make those names resolvable when loading from a script."""
    import __main__, inspect
    import framework
    for name, obj in vars(framework).items():
        if inspect.isclass(obj) and getattr(obj, "__module__", "") == "framework" and not hasattr(__main__, name):
            setattr(__main__, name, obj)


def load_artifact_and_metadata(dataset=None):
    pkl, meta = find_latest_artifact(dataset)
    _expose_framework_classes()
    with open(pkl, "rb") as f:
        artifact = pickle.load(f)
    if meta is not None:
        metadata = json.loads(meta.read_text(encoding="utf-8"))
        metadata["model_path"] = str(pkl)
    else:
        metadata = {"best_k": artifact["pipeline"].named_steps["selector"].k,
                    "best_params": artifact["pipeline"].named_steps["clf"].get_params(),
                    "model_path": str(pkl)}
    return artifact, metadata


# ---------------------------------------------------------------------------------
# VERBATIM helpers from the original notebook
# ---------------------------------------------------------------------------------

def build_safe_pipeline(model, k=None):
    """Build a baseline/ablation pipeline using the same safe preprocessing as the final model."""
    steps = [
        ('finite_before_imputer', SafeFiniteTransformer(inf_to_nan=True)),
        ('imputer', SimpleImputer(strategy='median')),
        ('finite_after_imputer', SafeFiniteTransformer()),
        ('scaler', RobustScaler(quantile_range=(1, 99))),
        ('finite_after_scaler', SafeFiniteTransformer()),
    ]
    if k is not None:
        steps += [
            ('selector', SelectKBest(score_func=f_classif, k=k)),
            ('finite_after_selector', SafeFiniteTransformer()),
        ]
    steps += [('clf', model)]
    return Pipeline(steps)


def load_publishability_data():
    """Load data using the same column handling as the optimizer."""
    X, y, le, numeric_cols, _shape = load_and_prepare_dataframe(DATASET_PATH)
    return X, y, le, numeric_cols


def same_split(X, y):
    """Recreate the optimizer's train/validation/test split for same-split checks."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VAL_SIZE, stratify=y_train_val, random_state=SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, pd.concat([X_train, X_val]), np.concatenate([y_train, y_val])


def metric_row(name, y_true, y_pred, extra=None):
    """Compact metric row for tables."""
    row = {
        'model': name,
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
    }
    if extra:
        row.update(extra)
    return row


def tree_resource_summary(pipe_or_tree):
    """Return DT deployability proxies when a fitted DecisionTreeClassifier is available."""
    tree_model = None
    if hasattr(pipe_or_tree, 'named_steps') and 'clf' in pipe_or_tree.named_steps:
        if isinstance(pipe_or_tree.named_steps['clf'], DecisionTreeClassifier):
            tree_model = pipe_or_tree.named_steps['clf']
    elif isinstance(pipe_or_tree, DecisionTreeClassifier):
        tree_model = pipe_or_tree
    if tree_model is None or not hasattr(tree_model, 'tree_'):
        return {'node_count': np.nan, 'leaf_count': np.nan, 'depth': np.nan, 'size_proxy_kb': np.nan}
    return {
        'node_count': int(tree_model.tree_.node_count),
        'leaf_count': int(tree_model.get_n_leaves()),
        'depth': int(tree_model.get_depth()),
        'size_proxy_kb': float(tree_model.tree_.node_count * EMBEDDED_BYTES_PER_NODE_EST / 1024.0),
    }


def main_params_for_model(model):
    """Short hyper-parameter summary instead of dumping every sklearn default."""
    keys = [
        'criterion', 'splitter', 'max_depth', 'min_samples_split', 'min_samples_leaf',
        'min_weight_fraction_leaf', 'max_features', 'class_weight', 'ccp_alpha',
        'n_estimators', 'learning_rate', 'max_iter', 'max_leaf_nodes', 'C', 'n_neighbors',
        'strategy', 'random_state'
    ]
    params = model.get_params() if hasattr(model, 'get_params') else {}
    return {k: params.get(k) for k in keys if k in params}


def clean_dt_params(params):
    """Remove duplicate random_state from saved params before constructing a new DecisionTreeClassifier."""
    p = dict(params)
    p.pop('random_state', None)
    return p


def evaluate_model(name, pipe, X_train, y_train, X_test, y_test, metadata_extra=None):
    """Fit a model and report metrics plus wall-clock training/prediction time."""
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = pipe.predict(X_test)
    pred_s = time.perf_counter() - t0
    extra = {'train_seconds': train_s, 'predict_seconds': pred_s}
    extra.update(tree_resource_summary(pipe))
    if metadata_extra:
        extra.update(metadata_extra)
    return metric_row(name, y_test, pred, extra), pred, pipe


def save_bar_plot(df, metric, title, filename, top_n=None):
    """Save a sorted horizontal bar plot and return its path."""
    plot_df = df.sort_values(metric, ascending=True)
    if top_n:
        plot_df = plot_df.tail(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(plot_df))))
    colors = ['#1f77b4' if 'Optimized' in str(m) or 'saved' in str(m).lower() else '#7f7f7f' for m in plot_df['model']]
    ax.barh(plot_df['model'], plot_df[metric], color=colors)
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    out = PLOTS_DIR / filename
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    return out


def save_scatter_plot(df, x, y, title, filename):
    """Save a scatter plot with model labels."""
    plot_df = df.dropna(subset=[x, y]).copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(plot_df[x], plot_df[y], s=80)
    for _, r in plot_df.iterrows():
        ax.annotate(r['model'], (r[x], r[y]), fontsize=8, xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = PLOTS_DIR / filename
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    return out
