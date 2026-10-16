"""
feature_policy.py -- helpers for notebook 07 (feature-policy / leakage ablation).

Goal: measure how much of a model's score comes from
  * identifier-like columns (source/destination PORTS),
  * traffic features that the substring pattern "ip" also removes
    (e.g. MIN_IP_PKT_LEN, MAX_IP_PKT_LEN, ICMP_IPV4_TYPE),
without loading the whole CSV: the train/validation/test partition is re-created from the LABELS
only (identical to the main pipeline, because sklearn's stratified split depends only on n and y),
and only the rows that are actually needed are read, chunk by chunk.
"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from framework import (_normalize_col, LEAKY_COLUMN_PATTERNS, DROP_COLS_IF_PRESENT, IP_COLS_IF_PRESENT,
                       TARGET_COL, TEST_SIZE, VAL_SIZE, SEED, stratified_sample)

_TIME_RE = re.compile(r"timestamp|flowstart|flowend|starttime|endtime")
_SRC_RE = re.compile(r"(^|[^a-z])(src|source)|^l4src|srcport|sourceport")
_DST_RE = re.compile(r"(^|[^a-z])(dst|dest|destination)|^l4dst|dstport|destport")


def classify_columns(header, target_col=TARGET_COL, drop_cols=None, ip_cols=None):
    """Split the CSV header into policy groups. Returns a dict of lists (original column names).
    drop_cols / ip_cols default to the ACTIVE profile; pass another profile's lists to classify a different dataset (cross-dataset tests)."""
    g = {"target": [], "label_leak": [], "identifiers": [], "time": [], "ports_src": [], "ports_dst": [], "ports_other": [],
         "ip_pattern_only": [], "core": []}
    drop_cols = DROP_COLS_IF_PRESENT if drop_cols is None else drop_cols
    ip_cols = IP_COLS_IF_PRESENT if ip_cols is None else ip_cols
    explicit = set(drop_cols) | set(ip_cols)
    for c in header:
        n = _normalize_col(c)
        if n == _normalize_col(target_col):
            g["target"].append(c)
        elif c in drop_cols:
            g["label_leak"].append(c)                    # e.g. binary 'Label' (would leak the target)
        elif "port" in n:                                  # source / destination port
            (g["ports_src"] if _SRC_RE.search(n) else g["ports_dst"] if _DST_RE.search(n) else g["ports_other"]).append(c)
        elif _TIME_RE.search(n):
            g["time"].append(c)                           # absolute time stamps (always dropped)
        elif c in explicit or "flowid" in n or "mac" in n:
            g["identifiers"].append(c)                    # IP addresses, flow ids, MACs (always dropped)
        elif any(p in n for p in LEAKY_COLUMN_PATTERNS):
            g["ip_pattern_only"].append(c)                # dropped ONLY because the substring 'ip' matched -> candidate legit features
        else:
            g["core"].append(c)
    return g


def read_target(path, target_col):
    """Read only the target column and return (LabelEncoder, y as int array in file order)."""
    s = pd.read_csv(path, usecols=[target_col], dtype={target_col: "category"})[target_col]
    cats = s.cat.categories.astype(str)
    le = LabelEncoder().fit(cats)
    mapping = le.transform(cats)
    codes = s.cat.codes.to_numpy()
    if (codes < 0).any():
        raise ValueError("missing values in the target column")
    return le, mapping[codes].astype(np.int64)


def make_partition(y, train_pool, eval_limit, seed=SEED):
    """Identical partition to eval_helpers.same_split, then capped stratified subsets of it.
    Returns dict of global row indices: pool (train), val, test, plus the label arrays aligned to them."""
    idx = np.arange(len(y))
    tv_idx, te_idx, y_tv, y_te = train_test_split(idx, y, test_size=TEST_SIZE, stratify=y, random_state=seed)
    tr_idx, va_idx, y_tr, y_va = train_test_split(tv_idx, y_tv, test_size=VAL_SIZE, stratify=y_tv, random_state=seed)
    pool_idx, _ = stratified_sample(tr_idx, y_tr, train_pool, seed)
    va_sel, _ = stratified_sample(va_idx, y_va, eval_limit, seed)
    te_sel, _ = stratified_sample(te_idx, y_te, eval_limit, seed)
    return {"pool": np.sort(pool_idx), "val": np.sort(va_sel), "test": np.sort(te_sel), "train_all": tr_idx}


def read_rows(path, columns, partition, chunksize=1_000_000):
    """Read ONLY the rows in partition['pool'|'val'|'test'] and ONLY `columns` (float32, inf -> NaN)."""
    n = max(int(max(v.max() for k, v in partition.items() if k in ("pool", "val", "test") and len(v))) + 1, 0)
    role = np.zeros(n, dtype=np.int8)
    role[partition["pool"]] = 1; role[partition["val"]] = 2; role[partition["test"]] = 3
    dtype_map = {c: "float32" for c in columns}
    try:
        it = pd.read_csv(path, usecols=columns, dtype=dtype_map, chunksize=chunksize)
        first = True
    except (ValueError, TypeError):
        it = pd.read_csv(path, usecols=columns, chunksize=chunksize)
    parts = {1: [], 2: [], 3: []}
    offset = 0
    for chunk in it:
        if offset >= n:
            break
        r = role[offset:offset + len(chunk)]
        if len(r) < len(chunk):
            r = np.concatenate([r, np.zeros(len(chunk) - len(r), dtype=np.int8)])
        for k in (1, 2, 3):
            m = r == k
            if m.any():
                sub = chunk.loc[m, columns].apply(pd.to_numeric, errors="coerce").astype("float32")
                sub.index = np.flatnonzero(m) + offset
                parts[k].append(sub)
        offset += len(chunk)
    out = {}
    for k, name in ((1, "pool"), (2, "val"), (3, "test")):
        if not parts[k]:
            out[name] = pd.DataFrame(columns=columns, dtype="float32"); continue      # empty role (e.g. no validation rows requested)
        df = pd.concat(parts[k]).sort_index()
        df = df.replace([np.inf, -np.inf], np.nan)
        out[name] = df
    return out


def numeric_columns(path, candidate_cols, nrows=50_000):
    """Which of the candidate columns are numeric (decided on the first rows, like the main loader's dtype check)."""
    if not candidate_cols:
        return []
    sample = pd.read_csv(path, usecols=candidate_cols, nrows=nrows)
    return sample.select_dtypes(include=[np.number]).columns.tolist()
