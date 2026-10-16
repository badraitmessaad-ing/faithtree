"""
generalization.py -- helpers for notebooks 12 and 14: does the model still work when the evaluation is NOT a random split of flows?

Random stratified splits of flow records can put near-duplicate flows (same host, same time window, same attack run) in both train and test,
which flatters every model. These helpers build harder, still leakage-free splits WITHOUT loading the whole CSV:
  * unseen-host split : all flows of a source IP fall in ONE partition (60/20/20 of hashed source-IP buckets) -> the test hosts were never seen in training
  * time split        : train on the first 60 % of the capture, test on the last 20 % (datasets that carry a timestamp)
  * cross-dataset     : train on one testbed, test on another (benign vs attack, shared NetFlow features) -- notebook 14
Partitions are dicts of global row positions {"pool", "val", "test"} that feature_policy.read_rows() understands.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SRC_IP_COL = {"nf_ton_iot_v2": "IPV4_SRC_ADDR", "nf_unsw_nb15_v3": "IPV4_SRC_ADDR", "cic_ton_iot": "Src IP"}
TIME_COL = {"nf_ton_iot_v2": None, "nf_unsw_nb15_v3": "FLOW_START_MILLISECONDS", "cic_ton_iot": "Timestamp"}


def base_name(ds):
    return ds.replace("_binary", "")


def _sample(idx, y_sub, n, seed):
    """<= n indices, stratified when possible (falls back to a plain random sample if a class has a single row)."""
    idx = np.asarray(idx)
    if len(idx) <= n:
        return idx
    try:
        keep, _ = train_test_split(idx, train_size=n, stratify=y_sub, random_state=seed)
        return keep
    except ValueError:
        return np.random.default_rng(seed).choice(idx, n, replace=False)


def read_hash_buckets(path, col, nbuckets=1000, chunksize=2_000_000):
    """Bucket id (0..nbuckets-1) of the hashed value of `col` for every row; keeps memory small (2 bytes per row)."""
    out = []
    for ch in pd.read_csv(path, usecols=[col], chunksize=chunksize, dtype={col: "string"}):
        h = pd.util.hash_pandas_object(ch[col].fillna(""), index=False).to_numpy()
        out.append((h % nbuckets).astype(np.uint16))
    return np.concatenate(out)


def read_time_values(path, col, chunksize=2_000_000):
    """Numeric time stamp per row (float seconds or ms; NaN when unparsable)."""
    out = []
    for ch in pd.read_csv(path, usecols=[col], chunksize=chunksize):
        s = ch[col]
        if pd.api.types.is_numeric_dtype(s):
            out.append(s.to_numpy(dtype="float64"))
        else:
            out.append(pd.to_datetime(s, errors="coerce").astype("int64").to_numpy(dtype="float64") / 1e9)
            out[-1][out[-1] < 0] = np.nan                               # NaT
    return np.concatenate(out)


def partition_by_buckets(y, buckets, pool_n, eval_n, seed=42):
    b = np.asarray(buckets).astype(np.int64) % 10
    tr, va, te = np.flatnonzero(b < 6), np.flatnonzero((b == 6) | (b == 7)), np.flatnonzero(b >= 8)
    return {"pool": np.sort(_sample(tr, y[tr], pool_n, seed)), "val": np.sort(va[:1000]), "test": np.sort(_sample(te, y[te], eval_n, seed)),
            "n_train_rows": len(tr), "n_test_rows": len(te)}


def partition_by_time(y, t, pool_n, eval_n, seed=42):
    ok = ~np.isnan(t); q60, q80 = np.nanquantile(t, [.6, .8])
    tr, te = np.flatnonzero(ok & (t <= q60)), np.flatnonzero(ok & (t > q80))
    return {"pool": np.sort(_sample(tr, y[tr], pool_n, seed)), "val": np.sort(np.flatnonzero(ok & (t > q60) & (t <= q80))[:1000]), "test": np.sort(_sample(te, y[te], eval_n, seed)),
            "n_train_rows": len(tr), "n_test_rows": len(te)}


def classes_missing_from_train(y_train, y_test):
    """Test classes that never occur in training (with their number of test rows)."""
    tr = set(np.unique(y_train).tolist()); u, c = np.unique(y_test, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c) if int(k) not in tr}


# ---------------------------------------------------------------------------------------------------------------------
# Notebook 12 helpers: everything is done on one uniform in-memory sample of rows, so many splits/ablations are cheap
# ---------------------------------------------------------------------------------------------------------------------
def bucket_masks(buckets, seed, nbuckets=1000, frac_train=0.6, frac_val=0.2):
    """Boolean masks (train, val, test) over rows: every flow of a hashed source-IP bucket goes to ONE partition.
    The bucket -> partition assignment is a random permutation (seed), so repeated seeds give different host splits."""
    rng = np.random.default_rng(seed)
    rank = np.empty(nbuckets, dtype=np.int64); rank[rng.permutation(nbuckets)] = np.arange(nbuckets)
    r = rank[np.asarray(buckets).astype(np.int64) % nbuckets]
    a, b = int(nbuckets * frac_train), int(nbuckets * (frac_train + frac_val))
    return r < a, (r >= a) & (r < b), r >= b


def class_host_table(y, buckets, class_names, train_mask, test_mask):
    """Per class: rows, distinct host buckets (a lower bound on distinct source IPs when < ~100), share of the busiest bucket,
    and how many of those buckets landed in train / test under the given host split."""
    y = np.asarray(y); b = np.asarray(buckets).astype(np.int64); rows = []
    for c in range(len(class_names)):
        m = y == c
        if not m.any():
            continue
        bc, cnt = np.unique(b[m], return_counts=True)
        tr_b = len(np.unique(b[m & train_mask])); te_b = len(np.unique(b[m & test_mask]))
        rows.append({"class": class_names[c], "rows": int(m.sum()), "distinct_host_buckets": int(len(bc)), "busiest_bucket_share": float(cnt.max() / m.sum()),
                     "buckets_in_train": tr_b, "buckets_in_test": te_b, "test_rows": int((m & test_mask).sum()), "train_rows": int((m & train_mask).sum())})
    return pd.DataFrame(rows)


def safe_sample(idx, y_sub, n, seed):
    return _sample(idx, y_sub, n, seed)
