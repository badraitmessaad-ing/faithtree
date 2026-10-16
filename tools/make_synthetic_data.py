"""
make_synthetic_data.py -- creates SMALL synthetic stand-ins for the three real CSVs
(same file names, same identifier/target column conventions, heavy class imbalance,
NaN/Inf values) for a dry run of the whole pipeline in minutes:

    python tools/make_synthetic_data.py --out data_synthetic
    set PAPER_DATA_DIR=data_synthetic   (Windows)   /   export PAPER_DATA_DIR=data_synthetic
    set PAPER_MODE=smoke
    python run_all.py

Results on synthetic data have no scientific meaning; they only show that every notebook,
figure and export step runs.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

TON = ["Benign", "backdoor", "ddos", "dos", "injection", "mitm", "password", "ransomware", "scanning", "xss"]
UNSW = ["Analysis", "Backdoor", "Benign", "DoS", "Exploits", "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms"]
WEIGHTS = np.array([0.80, 0.07, 0.045, 0.03, 0.02, 0.014, 0.010, 0.006, 0.003, 0.002])


def _features(n, n_feat, seed):
    X, y = make_classification(n_samples=n, n_features=n_feat, n_informative=int(n_feat * .55),
                               n_redundant=int(n_feat * .15), n_classes=10, n_clusters_per_class=2,
                               weights=WEIGHTS / WEIGHTS.sum(), class_sep=1.3, flip_y=0.01, random_state=seed)
    rng = np.random.default_rng(seed)
    # make it look like NetFlow: heavy tails + integer-like columns
    for j in range(0, n_feat, 3):
        X[:, j] = np.expm1(np.abs(X[:, j]) * 1.5) * 10
    for j in range(1, n_feat, 5):
        X[:, j] = np.round(np.abs(X[:, j]) * 50)
    X = X.astype(np.float64)
    # sprinkle NaN / +-Inf
    mask = rng.random(X.shape) < 0.002
    X[mask] = np.nan
    inf_mask = rng.random(X.shape) < 0.0005
    X[inf_mask] = np.inf
    return X, y


def make(out_dir, rows, seed=0):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ---- NF-ToN-IoT-v2 -----------------------------------------------------------
    X, y = _features(rows, 40, seed + 1)
    names = [f"FEAT_{i:02d}" for i in range(40)]
    df = pd.DataFrame(X, columns=names)
    df.insert(0, "IPV4_SRC_ADDR", [f"192.168.{a}.{b}" for a, b in rng.integers(0, 255, (rows, 2))])
    df.insert(1, "L4_SRC_PORT", rng.integers(1024, 65535, rows))
    df.insert(2, "IPV4_DST_ADDR", [f"10.0.{a}.{b}" for a, b in rng.integers(0, 255, (rows, 2))])
    df.insert(3, "L4_DST_PORT", rng.integers(1, 1024, rows))
    att = np.array(TON)[y]
    df["Label"] = (att != "Benign").astype(int)
    df["Attack"] = att
    df.to_csv(out / "NF-ToN-IoT-v2.csv", index=False)

    # ---- CIC-ToN-IoT -----------------------------------------------------------
    X, y = _features(rows, 50, seed + 2)
    names = [f"CIC_METRIC_{i:02d}" for i in range(50)]
    df = pd.DataFrame(X, columns=names)
    df.insert(0, "Flow ID", [f"f{i}" for i in range(rows)])
    df.insert(1, "Src IP", [f"192.168.{a}.{b}" for a, b in rng.integers(0, 255, (rows, 2))])
    df.insert(2, "Src Port", rng.integers(1024, 65535, rows))
    df.insert(3, "Dst IP", [f"10.0.{a}.{b}" for a, b in rng.integers(0, 255, (rows, 2))])
    df.insert(4, "Dst Port", rng.integers(1, 1024, rows))
    df.insert(5, "Timestamp", pd.date_range("2026-01-01", periods=rows, freq="s").astype(str))
    att = np.array(TON)[y]
    df["Label"] = (att != "Benign").astype(int)
    df["Attack"] = att
    df.to_csv(out / "CIC-ToN-IoT.csv", index=False)

    # ---- NF-UNSW-NB15-v3 -------------------------------------------------------
    X, y = _features(rows, 44, seed + 3)
    names = [f"NETFLOW_{i:02d}" for i in range(44)]
    df = pd.DataFrame(X, columns=names)
    df.insert(0, "IPV4_SRC_ADDR", [f"192.168.{a}.{b}" for a, b in rng.integers(0, 255, (rows, 2))])
    df.insert(1, "L4_SRC_PORT", rng.integers(1024, 65535, rows))
    df.insert(2, "IPV4_DST_ADDR", [f"10.0.{a}.{b}" for a, b in rng.integers(0, 255, (rows, 2))])
    df.insert(3, "L4_DST_PORT", rng.integers(1, 1024, rows))
    df.insert(4, "FLOW_START_MILLISECONDS", np.arange(rows) * 1000 + 1.6e12)
    df.insert(5, "FLOW_END_MILLISECONDS", np.arange(rows) * 1000 + 1.6e12 + 500)
    att = np.array(UNSW)[y]
    df["Label"] = (att != "Benign").astype(int)
    df["Attack"] = att
    df.to_csv(out / "NF-UNSW-NB15-v3.csv", index=False)
    print(f"wrote 3 synthetic CSVs ({rows:,} rows each) to {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_synthetic")
    ap.add_argument("--rows", type=int, default=30000)
    a = ap.parse_args()
    make(a.out, a.rows)
