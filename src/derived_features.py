"""
derived_features.py -- cheap per-flow features computed ONLY from other columns of the same row
(no leakage: no labels, no cross-row statistics, no identifiers).  Every feature is a ratio / sum / difference of
NetFlow counters, i.e. a handful of arithmetic operations -- cheap enough to compute on an ESP32 before the tree.

Used by notebook 10 to measure whether such features raise the MCC.  Columns that do not exist in a dataset are skipped
(e.g. the CICFlowMeter profile uses different column names -> nothing is added there).
The formulas are listed in FORMULAS so a later C export can reproduce them exactly.
"""
import numpy as np
import pandas as pd

FORMULAS = {
    "TOTAL_BYTES":        "IN_BYTES + OUT_BYTES",
    "TOTAL_PKTS":         "IN_PKTS + OUT_PKTS",
    "MEAN_PKT_SIZE":      "TOTAL_BYTES / (TOTAL_PKTS + 1)",
    "BYTES_PER_PKT_IN":   "IN_BYTES / (IN_PKTS + 1)",
    "BYTES_PER_PKT_OUT":  "OUT_BYTES / (OUT_PKTS + 1)",
    "PKT_RATIO_IN_OUT":   "IN_PKTS / (OUT_PKTS + 1)",
    "BYTE_RATIO_IN_OUT":  "IN_BYTES / (OUT_BYTES + 1)",
    "BYTES_PER_MS":       "TOTAL_BYTES / (FLOW_DURATION_MILLISECONDS + 1)",
    "PKTS_PER_MS":        "TOTAL_PKTS / (FLOW_DURATION_MILLISECONDS + 1)",
    "SMALL_PKT_FRAC":     "NUM_PKTS_UP_TO_128_BYTES / (TOTAL_PKTS + 1)",
    "LARGE_PKT_FRAC":     "NUM_PKTS_1024_TO_1514_BYTES / (TOTAL_PKTS + 1)",
    "TTL_RANGE":          "MAX_TTL - MIN_TTL",
    "WIN_RATIO":          "TCP_WIN_MAX_IN / (TCP_WIN_MAX_OUT + 1)",
    "RETRANS_FRAC_IN":    "RETRANSMITTED_IN_PKTS / (IN_PKTS + 1)",
    "RETRANS_FRAC_OUT":   "RETRANSMITTED_OUT_PKTS / (OUT_PKTS + 1)",
}


def add_derived_features(X: pd.DataFrame):
    """Return (X_with_extra_columns, list_of_new_column_names). Missing source columns -> the feature is skipped."""
    has = lambda *c: all(x in X.columns for x in c)
    f = lambda c: X[c].astype("float64")
    new = {}
    if has("IN_BYTES", "OUT_BYTES"):
        new["TOTAL_BYTES"] = f("IN_BYTES") + f("OUT_BYTES")
        new["BYTE_RATIO_IN_OUT"] = f("IN_BYTES") / (f("OUT_BYTES") + 1)
    if has("IN_PKTS", "OUT_PKTS"):
        new["TOTAL_PKTS"] = f("IN_PKTS") + f("OUT_PKTS")
        new["PKT_RATIO_IN_OUT"] = f("IN_PKTS") / (f("OUT_PKTS") + 1)
    if "TOTAL_BYTES" in new and "TOTAL_PKTS" in new:
        new["MEAN_PKT_SIZE"] = new["TOTAL_BYTES"] / (new["TOTAL_PKTS"] + 1)
    if has("IN_BYTES", "IN_PKTS"):
        new["BYTES_PER_PKT_IN"] = f("IN_BYTES") / (f("IN_PKTS") + 1)
    if has("OUT_BYTES", "OUT_PKTS"):
        new["BYTES_PER_PKT_OUT"] = f("OUT_BYTES") / (f("OUT_PKTS") + 1)
    if "FLOW_DURATION_MILLISECONDS" in X.columns:
        d = f("FLOW_DURATION_MILLISECONDS") + 1
        if "TOTAL_BYTES" in new: new["BYTES_PER_MS"] = new["TOTAL_BYTES"] / d
        if "TOTAL_PKTS" in new: new["PKTS_PER_MS"] = new["TOTAL_PKTS"] / d
    if "TOTAL_PKTS" in new:
        if "NUM_PKTS_UP_TO_128_BYTES" in X.columns: new["SMALL_PKT_FRAC"] = f("NUM_PKTS_UP_TO_128_BYTES") / (new["TOTAL_PKTS"] + 1)
        if "NUM_PKTS_1024_TO_1514_BYTES" in X.columns: new["LARGE_PKT_FRAC"] = f("NUM_PKTS_1024_TO_1514_BYTES") / (new["TOTAL_PKTS"] + 1)
    if has("MAX_TTL", "MIN_TTL"):
        new["TTL_RANGE"] = f("MAX_TTL") - f("MIN_TTL")
    if has("TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"):
        new["WIN_RATIO"] = f("TCP_WIN_MAX_IN") / (f("TCP_WIN_MAX_OUT") + 1)
    if has("RETRANSMITTED_IN_PKTS", "IN_PKTS"):
        new["RETRANS_FRAC_IN"] = f("RETRANSMITTED_IN_PKTS") / (f("IN_PKTS") + 1)
    if has("RETRANSMITTED_OUT_PKTS", "OUT_PKTS"):
        new["RETRANS_FRAC_OUT"] = f("RETRANSMITTED_OUT_PKTS") / (f("OUT_PKTS") + 1)
    if not new:
        return X, []
    extra = pd.DataFrame(new, index=X.index).replace([np.inf, -np.inf], np.nan).astype("float32")
    return pd.concat([X, extra], axis=1), list(extra.columns)
