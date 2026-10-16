"""
esp32_export.py -- export a fitted tree pipeline to C for the ESP32 and verify it.

Exact export (format version 3)
  raw_space_thresholds / verify_raw_thresholds   thresholds mapped into the raw float32 input space
  verify_edge_cases / verify_monotone            NaN, +/-inf, signed zero, extremes; monotonicity per feature
  check_node_format                              tree must fit the 10-byte node (int16 children, uint8 feature)
  export_tree_pipeline_to_c                      writes esp32_dt_model.h (no preprocessing on the device)
  emulate_c_predict                              numpy re-implementation of the generated C code
  mismatch_report                                rows where C logic and scikit-learn disagree

Legacy export (kept as a baseline for the paper)
  export_tree_pipeline_to_c_legacy / emulate_c_predict_legacy
                                                 float32 preprocessing on the device, thresholds
                                                 rounded to the nearest float32

Board and host tools
  select_validation_samples      which held-out rows are flashed ("first_n" or "class_balanced")
  create_test_vectors_header     test vectors as C arrays (NaN -> NAN, +/-Inf -> +/-INFINITY)
  write_benchmark_sketch         Arduino sketch that times only predict_esp32_dt()
  write_footprint_sketch         sketch pair for measuring the model's flash cost
  host_c_check / host_c_check_full   compile the header with a host C compiler and compare
"""
import json, subprocess, shutil, tempfile
from pathlib import Path
import numpy as np
import pandas as pd


def export_tree_pipeline_to_c_legacy(artifact, out_dir):
    """Legacy exporter: float32 preprocessing on the device and thresholds rounded to the nearest
    float32.  Kept as a baseline; it disagrees with scikit-learn when an input lands exactly on a
    threshold that rounding moved upwards (3 of 3,388,100 NF-ToN-IoT-v2 test rows)."""
    pipe = artifact["pipeline"]
    le = artifact["label_encoder"]
    numeric_cols = artifact["numeric_cols"]

    imputer = pipe.named_steps["imputer"]
    scaler = pipe.named_steps["scaler"]
    selector = pipe.named_steps["selector"]
    tree = pipe.named_steps["clf"].tree_

    selected_idx = selector.get_support(indices=True)

    h = [
        "#pragma once",
        "#include <stdint.h>",
        "#include <math.h>",
        f"#define N_RAW_FEATURES {len(numeric_cols)}",
        f"#define N_SELECTED_FEATURES {len(selected_idx)}",
        f"#define N_NODES {tree.node_count}",
        f"#define N_CLASSES {len(le.classes_)}",
    ]

    def arr(name, values, ctype="float"):
        h.append(
            f"static const {ctype} {name}[] = "
            + "{"
            + ",".join("%.9g" % float(v) for v in values)
            + "};"
        )

    def arri(name, values, ctype="int16_t"):
        h.append(
            f"static const {ctype} {name}[] = "
            + "{"
            + ",".join(str(int(v)) for v in values)
            + "};"
        )

    # IMPORTANT:
    # In this pipeline the order is:
    #   imputer -> scaler -> selector -> decision tree
    # Therefore imputer/scaler arrays are originally indexed by raw feature index,
    # while tree.feature is indexed by selected-feature index.
    # We export only selected imputer/scaler values so every ESP32 preprocessing
    # array uses selected-feature index j = 0..N_SELECTED_FEATURES-1.
    arr("IMPUTER_MEDIAN_SELECTED", imputer.statistics_[selected_idx])
    arr("SCALER_CENTER_SELECTED", scaler.center_[selected_idx])
    arr("SCALER_SCALE_SELECTED", scaler.scale_[selected_idx])

    arri("SELECTED_IDX", selected_idx, "uint8_t")
    arri("CHILD_LEFT", tree.children_left, "int16_t")
    arri("CHILD_RIGHT", tree.children_right, "int16_t")
    arri("FEATURE", tree.feature, "int16_t")
    arr("THRESHOLD", tree.threshold)

    values = tree.value.reshape(tree.node_count, -1)
    node_class = np.argmax(values, axis=1)
    arri("NODE_CLASS", node_class, "uint8_t")

    h.append(
        "static const char* CLASS_NAMES[] = {"
        + ",".join(json.dumps(str(c)) for c in le.classes_)
        + "};"
    )

    h.append(r'''
static inline float safe_value(float x) {
  if (!isfinite(x)) return 0.0f;
  if (x > 1e10f) return 1e10f;
  if (x < -1e10f) return -1e10f;
  return x;
}

int predict_esp32_dt(const float raw[N_RAW_FEATURES]) {
  float selected[N_SELECTED_FEATURES];

  for (int j = 0; j < N_SELECTED_FEATURES; ++j) {
    int i = SELECTED_IDX[j];

    float x = raw[i];

    if (!isfinite(x)) {
      x = IMPUTER_MEDIAN_SELECTED[j];
    }

    x = safe_value(x);
    x = (x - SCALER_CENTER_SELECTED[j]) / SCALER_SCALE_SELECTED[j];
    x = safe_value(x);

    selected[j] = x;
  }

  int node = 0;

  while (CHILD_LEFT[node] != -1) {
    int f = FEATURE[node];

    if (selected[f] <= THRESHOLD[node]) {
      node = CHILD_LEFT[node];
    } else {
      node = CHILD_RIGHT[node];
    }
  }

  return NODE_CLASS[node];
}
''')

    out_path = Path(out_dir) / "esp32_dt_model.h"
    out_path.write_text("\n".join(h), encoding="utf-8")
    return out_path




# -----------------------------------------------------------------------------
# Which held-out rows go on the board
# -----------------------------------------------------------------------------
def select_validation_samples(y_test, n=5000, mode="class_balanced", seed=42):
    """Return sorted positional indices into the test split.

    first_n        : the first n rows (what the earlier deployment used; mostly majority class)
    class_balanced : up to n//n_classes rows per class (all rows of rare classes), then the
                     remaining slots are filled at random -> every class exercises the C code.
    """
    y = np.asarray(y_test)
    n = min(n, len(y))
    if mode == "first_n":
        return np.arange(n)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    cap = max(1, n // len(classes))
    chosen = []
    for c in classes:
        idx = np.flatnonzero(y == c)
        take = idx if len(idx) <= cap else rng.choice(idx, cap, replace=False)
        chosen.extend(take.tolist())
    chosen = np.array(sorted(set(chosen)))
    if len(chosen) < n:
        rest = np.setdiff1d(np.arange(len(y)), chosen)
        fill = rng.choice(rest, min(n - len(chosen), len(rest)), replace=False)
        chosen = np.sort(np.concatenate([chosen, fill]))
    return chosen[:n] if len(chosen) > n else chosen


def _c_float(v):
    v = float(v)
    if np.isnan(v):
        return "NAN"
    if np.isposinf(v):
        return "INFINITY"
    if np.isneginf(v):
        return "-INFINITY"
    return "%.9g" % v


def create_test_vectors_header(X_sample, y_true, y_pred, artifact, out_dir, sample_ids=None):
    """Write the test vectors as a C header; NaN and +/-Inf become NAN and +/-INFINITY."""
    numeric_cols = artifact["numeric_cols"]
    X_np = X_sample[numeric_cols].to_numpy(dtype=np.float32) if hasattr(X_sample, "columns") \
        else np.asarray(X_sample, dtype=np.float32)
    ids = np.arange(len(X_np), dtype=np.uint32) if sample_ids is None else np.asarray(sample_ids, dtype=np.uint32)
    lines = [
        "#pragma once", "#include <stdint.h>", "#include <math.h>", "",
        f"#define TEST_SAMPLE_COUNT {X_np.shape[0]}",
        f"#define TEST_FEATURE_COUNT {X_np.shape[1]}", "",
        "static const uint32_t TEST_SAMPLE_ID[TEST_SAMPLE_COUNT] = {" + ",".join(str(int(i)) for i in ids) + "};", "",
        "static const uint8_t TEST_TRUE_LABEL[TEST_SAMPLE_COUNT] = {" + ",".join(str(int(v)) for v in y_true) + "};", "",
        "static const uint8_t TEST_PYTHON_PRED[TEST_SAMPLE_COUNT] = {" + ",".join(str(int(v)) for v in y_pred) + "};", "",
        "static const float TEST_X[TEST_SAMPLE_COUNT][TEST_FEATURE_COUNT] = {",
    ]
    for row in X_np:
        lines.append("  {" + ",".join(_c_float(v) for v in row) + "},")
    lines.append("};")
    text = "\n".join(lines)
    out_path = Path(out_dir) / "esp32_test_vectors.h"
    out_path.write_text(text, encoding="utf-8")
    assert ",nan" not in text and ",inf" not in text, "invalid lowercase nan/inf token in header"
    return out_path


# -----------------------------------------------------------------------------
# Sketches
# -----------------------------------------------------------------------------
BENCHMARK_SKETCH = r"""
// =============================================================================
// esp32_dt_test.ino  --  Python-vs-ESP32 fidelity + inference-latency benchmark
//
// IMPORTANT (measurement hygiene):
//   The earlier version of this sketch started the timer BEFORE the loop that
//   also called Serial.print() for every sample, so the reported "us/sample"
//   was dominated by UART transmission time at 115200 baud (~87 us per
//   character), not by the decision tree.  This version times ONLY
//   predict_esp32_dt() and prints everything afterwards.
//
// Output (CSV-like, parsed by notebooks/05_esp32_hardware_analysis.ipynb):
//   #META,key,value ...
//   sample_id,true_label,python_pred,esp32_pred,match,infer_us
//   SUMMARY_MATCHES,a/b ... INFER_US_* ... #DONE
// =============================================================================
#include <Arduino.h>
#include <string.h>
#include <stdlib.h>
#include "esp32_dt_model.h"
#include "esp32_test_vectors.h"

#ifndef BATCH_REPS
#define BATCH_REPS 20          // full passes over all samples for the precise mean
#endif
#define WARMUP_SAMPLES 200

static uint16_t INFER_US[TEST_SAMPLE_COUNT];   // per-sample time (saturates at 65535 us)
static uint8_t  ESP_PRED[TEST_SAMPLE_COUNT];
volatile int    g_sink = 0;                    // stops the compiler removing the timed loop

static int cmp_u16(const void* a, const void* b) {
  return (int)(*(const uint16_t*)a) - (int)(*(const uint16_t*)b);
}

void setup() {
  Serial.begin(115200);
  delay(2000);

  uint32_t heap_before = ESP.getFreeHeap();

  // ---- 1. warm-up (instruction cache / flash cache) -------------------------
  int warm = TEST_SAMPLE_COUNT < WARMUP_SAMPLES ? TEST_SAMPLE_COUNT : WARMUP_SAMPLES;
  for (int i = 0; i < warm; i++) g_sink += predict_esp32_dt(TEST_X[i]);

  // ---- 2. per-sample timing (NO printing inside this loop) ------------------
  for (int i = 0; i < TEST_SAMPLE_COUNT; i++) {
    uint32_t t0 = micros();
    int p = predict_esp32_dt(TEST_X[i]);
    uint32_t dt = micros() - t0;
    ESP_PRED[i] = (uint8_t)p;
    INFER_US[i] = dt > 65535 ? 65535 : (uint16_t)dt;
  }

  // ---- 3. batch timing: precise mean over BATCH_REPS passes -----------------
  uint32_t tb0 = micros();
  for (int r = 0; r < BATCH_REPS; r++)
    for (int i = 0; i < TEST_SAMPLE_COUNT; i++) g_sink += predict_esp32_dt(TEST_X[i]);
  uint32_t tb = micros() - tb0;
  float mean_batch_us = (float)tb / ((float)TEST_SAMPLE_COUNT * BATCH_REPS);

  uint32_t heap_after = ESP.getFreeHeap();

  // ---- 4. statistics ---------------------------------------------------------
  int matches = 0, true_correct = 0;
  uint64_t sum_single = 0;
  for (int i = 0; i < TEST_SAMPLE_COUNT; i++) {
    if (ESP_PRED[i] == TEST_PYTHON_PRED[i]) matches++;
    if (ESP_PRED[i] == TEST_TRUE_LABEL[i])  true_correct++;
    sum_single += INFER_US[i];
  }
  static uint16_t sorted_us[TEST_SAMPLE_COUNT];
  memcpy(sorted_us, INFER_US, sizeof(sorted_us));
  qsort(sorted_us, TEST_SAMPLE_COUNT, sizeof(uint16_t), cmp_u16);

  // ---- 5. print everything (after all timing is finished) -------------------
  Serial.println("#META,begin");
  Serial.print("#META,chip_model,");        Serial.println(ESP.getChipModel());
  Serial.print("#META,chip_revision,");     Serial.println(ESP.getChipRevision());
  Serial.print("#META,cpu_mhz,");           Serial.println(ESP.getCpuFreqMHz());
  Serial.print("#META,sdk_version,");       Serial.println(ESP.getSdkVersion());
  Serial.print("#META,sketch_size_bytes,"); Serial.println(ESP.getSketchSize());
  Serial.print("#META,free_sketch_space_bytes,"); Serial.println(ESP.getFreeSketchSpace());
  Serial.print("#META,heap_size_bytes,");   Serial.println(ESP.getHeapSize());
  Serial.print("#META,free_heap_before,");  Serial.println(heap_before);
  Serial.print("#META,free_heap_after,");   Serial.println(heap_after);
  Serial.print("#META,min_free_heap,");     Serial.println(ESP.getMinFreeHeap());
  Serial.print("#META,n_nodes,");           Serial.println(N_NODES);
  Serial.print("#META,n_selected_features,"); Serial.println(N_SELECTED_FEATURES);
  Serial.print("#META,batch_reps,");        Serial.println(BATCH_REPS);
  Serial.println("#META,end");

  Serial.println("sample_id,true_label,python_pred,esp32_pred,match,infer_us");
  for (int i = 0; i < TEST_SAMPLE_COUNT; i++) {
    Serial.print(TEST_SAMPLE_ID[i]);   Serial.print(",");
    Serial.print(TEST_TRUE_LABEL[i]);  Serial.print(",");
    Serial.print(TEST_PYTHON_PRED[i]); Serial.print(",");
    Serial.print(ESP_PRED[i]);         Serial.print(",");
    Serial.print(ESP_PRED[i] == TEST_PYTHON_PRED[i] ? 1 : 0); Serial.print(",");
    Serial.println(INFER_US[i]);
  }

  Serial.print("SUMMARY_MATCHES,");      Serial.print(matches); Serial.print("/"); Serial.println(TEST_SAMPLE_COUNT);
  Serial.print("SUMMARY_TRUE_CORRECT,"); Serial.print(true_correct); Serial.print("/"); Serial.println(TEST_SAMPLE_COUNT);
  Serial.print("INFER_US_MEAN_BATCH,");  Serial.println(mean_batch_us, 4);
  Serial.print("INFER_US_MEAN_SINGLE,"); Serial.println((float)sum_single / TEST_SAMPLE_COUNT, 4);
  Serial.print("INFER_US_MIN,");         Serial.println(sorted_us[0]);
  Serial.print("INFER_US_MEDIAN,");      Serial.println(sorted_us[TEST_SAMPLE_COUNT / 2]);
  Serial.print("INFER_US_P95,");         Serial.println(sorted_us[(int)(TEST_SAMPLE_COUNT * 0.95)]);
  Serial.print("INFER_US_P99,");         Serial.println(sorted_us[(int)(TEST_SAMPLE_COUNT * 0.99)]);
  Serial.print("INFER_US_MAX,");         Serial.println(sorted_us[TEST_SAMPLE_COUNT - 1]);
  Serial.println("#DONE");
}

void loop() {}
"""

FOOTPRINT_SKETCH = r"""
// =============================================================================
// esp32_dt_footprint.ino -- measures the MODEL's own flash cost.
//
// Build twice and compare the "Sketch uses N bytes" line printed by the compiler:
//   1) USE_MODEL 1  -> firmware WITH the decision tree      = A bytes
//   2) USE_MODEL 0  -> same firmware WITHOUT the tree       = B bytes
//   model flash footprint = A - B
// (The benchmark sketch is much larger because it also embeds thousands of test
//  vectors; do not report its size as the model size.)
// =============================================================================
#define USE_MODEL 1

#include <Arduino.h>
#if USE_MODEL
#include "esp32_dt_model.h"
static const float ZERO_VEC[N_RAW_FEATURES] = {0};
#endif

volatile int g_sink = 0;

void setup() {
  Serial.begin(115200);
  delay(2000);
#if USE_MODEL
  uint32_t heap0 = ESP.getFreeHeap();
  g_sink += predict_esp32_dt(ZERO_VEC);
  uint32_t heap1 = ESP.getFreeHeap();
  Serial.print("MODEL_NODES,");       Serial.println(N_NODES);
  Serial.print("HEAP_DELTA_BYTES,");  Serial.println((int)heap0 - (int)heap1);
#endif
  Serial.print("SKETCH_SIZE_BYTES,"); Serial.println(ESP.getSketchSize());
  Serial.print("FREE_HEAP_BYTES,");   Serial.println(ESP.getFreeHeap());
  Serial.println("#DONE");
}
void loop() {}
"""


def write_benchmark_sketch(out_dir):
    p = Path(out_dir) / "esp32_dt_test.ino"
    p.write_text(BENCHMARK_SKETCH, encoding="utf-8")
    return p


def write_footprint_sketch(model_header, out_dir):
    d = Path(out_dir) / "esp32_dt_footprint"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "esp32_dt_footprint.ino"
    p.write_text(FOOTPRINT_SKETCH, encoding="utf-8")
    shutil.copy(Path(model_header), d / "esp32_dt_model.h")
    return p


# -----------------------------------------------------------------------------
# float32 emulation of the generated C predictor (works on any number of rows)
# -----------------------------------------------------------------------------
def emulate_c_predict_legacy(artifact, X_raw):
    """Re-implements predict_esp32_dt() in numpy float32 -> predictions for every row of X_raw."""
    pipe = artifact["pipeline"]
    numeric_cols = artifact["numeric_cols"]
    imputer, scaler = pipe.named_steps["imputer"], pipe.named_steps["scaler"]
    selector, tree = pipe.named_steps["selector"], pipe.named_steps["clf"].tree_
    sel = selector.get_support(indices=True)

    raw = (X_raw[numeric_cols].to_numpy(dtype=np.float32) if hasattr(X_raw, "columns")
           else np.asarray(X_raw, dtype=np.float32))
    x = raw[:, sel].copy()
    med = imputer.statistics_[sel].astype(np.float32)
    ctr = scaler.center_[sel].astype(np.float32)
    scl = scaler.scale_[sel].astype(np.float32)

    bad = ~np.isfinite(x)
    x = np.where(bad, np.broadcast_to(med, x.shape), x).astype(np.float32)

    def safe(v):
        v = np.where(np.isfinite(v), v, np.float32(0))
        return np.clip(v, np.float32(-1e10), np.float32(1e10)).astype(np.float32)

    x = safe(x)
    x = safe(((x - ctr) / scl).astype(np.float32))

    left, right = tree.children_left, tree.children_right
    feat = tree.feature
    thr = tree.threshold.astype(np.float32)
    node_class = np.argmax(tree.value.reshape(tree.node_count, -1), axis=1)

    node = np.zeros(len(x), dtype=np.int64)
    rows = np.arange(len(x))
    active = left[node] != -1
    while active.any():
        a = rows[active]
        n = node[a]
        go_left = x[a, feat[n]] <= thr[n]
        node[a] = np.where(go_left, left[n], right[n])
        active = left[node] != -1
    return node_class[node]


# -----------------------------------------------------------------------------
# Host-side compile-and-run check of the generated header (needs gcc/cc)
# -----------------------------------------------------------------------------
HOST_MAIN = r"""
#include <stdio.h>
#include "esp32_dt_model.h"
#include "esp32_test_vectors.h"
int main(void){
  for (int i = 0; i < TEST_SAMPLE_COUNT; i++)
    printf("%d\n", predict_esp32_dt(TEST_X[i]));
  return 0;
}
"""


# =============================================================================
# Exact export (format version 3): thresholds mapped into the raw float32 input space
# =============================================================================
# Motivation: the legacy export reproduced scikit-learn on 3,388,097 / 3,388,100 rows.  Two
# 1-ulp effects caused the 3 disagreements:
#   (a) preprocessing: on the device x' = (x - c) / float32(s); scikit-learn divides by the
#       float64 scale (two features have scales that are not float32 numbers), so x' can
#       differ by one ulp;
#   (b) thresholds: scikit-learn compares float32(x') <= t with t in float64; the legacy
#       header stored t rounded to the NEAREST float32, which is above t for ~1/4 of nodes.
# Method: every step between the raw feature and the comparison is monotone non-decreasing in
# the raw value (inf->NaN, median imputation, clipping, RobustScaler with scale>0, float32
# casts, feature selection).  So for every internal node there is a largest float32 raw
# value R such that scikit-learn sends x left  <=>  x <= R.  We find R by bisection over
# the ordered float32 bit patterns, evaluating the *fitted scikit-learn pipeline itself*,
# and store R.  The device then does no arithmetic at all:
#        x = raw[f];  if (!isfinite(x)) x = median[f];  go left iff x <= R[node]
# which agrees with scikit-learn for EVERY float32 input vector (not only the tested ones).
# -----------------------------------------------------------------------------
_FLT_MAX = np.float32(np.finfo(np.float32).max)


def _f32_to_key(v):
    u = np.asarray(v, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.where(u >= 0, u, -(u & 0x7FFFFFFF))


def _key_to_f32(k):
    k = np.asarray(k, dtype=np.int64)
    u = np.where(k >= 0, k, (-k) | 0x80000000).astype(np.uint32)
    return u.view(np.float32)


def _pre_tree_transform(pipe):
    """Every pipeline step before the classifier (imputer, scalers, selector, safety clips)."""
    return pipe[:-1]


def raw_space_thresholds(artifact):
    """Return (raw_feature[node], raw_threshold[node]) such that, for every finite float32 x,
    scikit-learn routes x left at `node` iff x <= raw_threshold[node].  Leaves get feature 0
    and threshold 0 (never used).  +inf / -inf mean 'always left' / 'always right'."""
    pipe = artifact["pipeline"]
    numeric_cols = list(artifact["numeric_cols"])
    tree = pipe.named_steps["clf"].tree_
    sel = pipe.named_steps["selector"].get_support(indices=True)
    median = pipe.named_steps["imputer"].statistics_.astype(np.float32)
    pre = _pre_tree_transform(pipe)

    internal = np.where(tree.children_left != -1)[0]
    f_sel = tree.feature[internal]
    f_raw = sel[f_sel]
    thr = tree.threshold[internal].astype(np.float64)
    m = len(internal)
    rows = np.arange(m)

    def goes_left(cand):
        M = np.tile(median, (m, 1))
        M[rows, f_raw] = cand
        Z = np.asarray(pre.transform(pd.DataFrame(M, columns=numeric_cols)))
        z = Z[rows, f_sel].astype(np.float32).astype(np.float64)   # the tree sees float32
        return z <= thr

    lo = np.full(m, _f32_to_key(-_FLT_MAX), dtype=np.int64)
    hi = np.full(m, _f32_to_key(_FLT_MAX), dtype=np.int64)
    left_at_max = goes_left(np.full(m, _FLT_MAX, np.float32))
    left_at_min = goes_left(np.full(m, -_FLT_MAX, np.float32))
    out = np.empty(m, dtype=np.float32)
    out[left_at_max] = np.inf
    out[~left_at_min & ~left_at_max] = -np.inf
    todo = left_at_min & ~left_at_max            # invariant: left at lo, right at hi
    while True:
        open_ = todo & (hi - lo > 1)
        if not open_.any():
            break
        mid = lo + (hi - lo) // 2
        cand = _key_to_f32(np.where(open_, mid, lo))
        gl = goes_left(cand)
        lo = np.where(open_ & gl, mid, lo)
        hi = np.where(open_ & ~gl, mid, hi)
    out[todo] = _key_to_f32(lo[todo])

    raw_feature = np.zeros(tree.node_count, dtype=np.int64)
    raw_thr = np.zeros(tree.node_count, dtype=np.float32)
    raw_feature[internal] = f_raw
    raw_thr[internal] = out
    return raw_feature, raw_thr


def verify_raw_thresholds(artifact, raw_feature, raw_thr):
    """Proof-style check against the fitted pipeline: at every internal node, R goes left and
    the next float32 above R goes right (so the cut is exactly where scikit-learn puts it)."""
    pipe = artifact["pipeline"]
    numeric_cols = list(artifact["numeric_cols"])
    tree = pipe.named_steps["clf"].tree_
    sel = pipe.named_steps["selector"].get_support(indices=True)
    inv = {int(r): j for j, r in enumerate(sel)}
    median = pipe.named_steps["imputer"].statistics_.astype(np.float32)
    pre = _pre_tree_transform(pipe)
    internal = np.where(tree.children_left != -1)[0]
    R = raw_thr[internal]
    fin = np.isfinite(R)
    idx, R = internal[fin], R[fin]
    f_raw = raw_feature[idx]
    f_sel = np.array([inv[int(r)] for r in f_raw])
    m = len(idx); rows = np.arange(m)

    def z_of(c):
        M = np.tile(median, (m, 1)); M[rows, f_raw] = c
        Z = np.asarray(pre.transform(pd.DataFrame(M, columns=numeric_cols)))
        return Z[rows, f_sel].astype(np.float32).astype(np.float64)    # the tree compares float32 values

    up = np.nextafter(R, np.float32(np.inf), dtype=np.float32)
    ok_at = z_of(R) <= tree.threshold[idx]
    ok_above = z_of(up) > tree.threshold[idx]
    return {"internal_nodes": int(len(internal)), "finite_cuts_checked": int(m),
            "always_left_or_right": int((~fin).sum()),
            "cut_goes_left": int(ok_at.sum()), "next_float_goes_right": int(ok_above.sum()),
            "exact": bool(ok_at.all() and ok_above.all())}


MAX_NODES = 32_767        # children are int16_t; -1 marks a leaf, so indices 0..32,766 are usable
MAX_RAW_FEATURES = 256    # RAW_FEATURE is uint8_t
MAX_CLASSES = 256         # NODE_CLASS is uint8_t


def check_node_format(n_nodes, n_raw_features, n_classes):
    """Refuse trees that do not fit the 10-byte node layout instead of writing a wrong header."""
    problems = []
    if n_nodes > MAX_NODES:
        problems.append(f"{n_nodes:,} nodes > {MAX_NODES:,} (int16_t child indices)")
    if n_raw_features > MAX_RAW_FEATURES:
        problems.append(f"{n_raw_features} raw features > {MAX_RAW_FEATURES} (uint8_t feature index)")
    if n_classes > MAX_CLASSES:
        problems.append(f"{n_classes} classes > {MAX_CLASSES} (uint8_t class label)")
    if problems:
        raise ValueError("tree does not fit the 10-byte node format: " + "; ".join(problems)
                         + ". Constrain the tree (max_depth, min_samples_leaf, ccp_alpha) or widen the arrays.")


def _route_python(artifact, f_raw, f_sel, thr, values):
    """scikit-learn routing at many nodes at once: True where the value goes left.
    All other raw features sit at their training median; only feature f_raw[i] changes."""
    pipe = artifact["pipeline"]
    median = pipe.named_steps["imputer"].statistics_.astype(np.float32)
    m = len(f_raw); rows = np.arange(m)
    M = np.tile(median, (m, 1)); M[rows, f_raw] = values
    Z = np.asarray(_pre_tree_transform(pipe).transform(pd.DataFrame(M, columns=list(artifact["numeric_cols"]))))
    return Z[rows, f_sel].astype(np.float32).astype(np.float64) <= thr


def verify_edge_cases(artifact, raw_feature, raw_thr):
    """Inputs that the bisection never visits, checked at every internal node (including nodes
    whose cut is +/-inf): NaN, +inf, -inf, -0.0, +0.0, -FLT_MAX and +FLT_MAX.  For each one the
    device rule  (x non-finite -> float32 median;  left iff x <= R)  must route like scikit-learn.
    Also checks that every imputer median is exactly a float32 number, so that storing it as
    float32 on the device does not move it."""
    pipe = artifact["pipeline"]
    tree = pipe.named_steps["clf"].tree_
    sel = pipe.named_steps["selector"].get_support(indices=True)
    inv = {int(r): j for j, r in enumerate(sel)}
    med64 = np.asarray(pipe.named_steps["imputer"].statistics_, dtype=np.float64)
    med32 = med64.astype(np.float32)
    internal = np.where(tree.children_left != -1)[0]
    f_raw = raw_feature[internal]
    f_sel = np.array([inv[int(r)] for r in f_raw])
    thr = tree.threshold[internal]
    R = raw_thr[internal].astype(np.float32)
    cases = {"nan": np.nan, "+inf": np.inf, "-inf": -np.inf, "-0.0": -0.0, "+0.0": 0.0,
             "-FLT_MAX": -_FLT_MAX, "+FLT_MAX": _FLT_MAX}
    out = {"internal_nodes": int(len(internal)),
           "medians_are_float32": bool(np.all(med32.astype(np.float64) == med64))}
    ok = out["medians_are_float32"]
    for name, v in cases.items():
        x = np.full(len(internal), v, dtype=np.float32)
        x_dev = np.where(np.isfinite(x), x, med32[f_raw])
        dev_left = x_dev <= R
        py_left = _route_python(artifact, f_raw, f_sel, thr, x)
        agree = int((dev_left == py_left).sum())
        out[name] = agree
        ok = ok and agree == len(internal)
    out["exact"] = bool(ok)
    return out


def verify_monotone(artifact, raw_feature, n_points=2049):
    """Spot-check of the monotonicity assumption: for every raw feature used by the tree, the
    value that reaches the tree must be non-decreasing over n_points float32 values spread
    evenly over the ordered float32 range (plus -0.0 and +0.0)."""
    pipe = artifact["pipeline"]
    tree = pipe.named_steps["clf"].tree_
    sel = pipe.named_steps["selector"].get_support(indices=True)
    inv = {int(r): j for j, r in enumerate(sel)}
    median = pipe.named_steps["imputer"].statistics_.astype(np.float32)
    used = sorted(set(int(f) for f in raw_feature[tree.children_left != -1]))
    keys = np.linspace(_f32_to_key(-_FLT_MAX), _f32_to_key(_FLT_MAX), n_points).astype(np.int64)
    xs = np.unique(np.concatenate([_key_to_f32(keys), np.array([-0.0, 0.0], np.float32)]))
    xs = xs[np.argsort(_f32_to_key(xs), kind="stable")]
    bad = []
    for f in used:
        M = np.tile(median, (len(xs), 1)); M[:, f] = xs
        Z = np.asarray(_pre_tree_transform(pipe).transform(pd.DataFrame(M, columns=list(artifact["numeric_cols"]))))
        z = Z[:, inv[f]].astype(np.float32)
        if np.any(np.diff(z.astype(np.float64)) < 0):
            bad.append(f)
    return {"features_checked": len(used), "points_per_feature": int(len(xs)),
            "non_monotone_features": bad, "exact": not bad}


def _c_hex(v):
    v = float(v)
    if np.isnan(v):
        return "NAN"
    if np.isposinf(v):
        return "INFINITY"
    if np.isneginf(v):
        return "-INFINITY"
    # shortest decimal that round-trips the (float32-valued) double exactly; valid in C and C++11
    return repr(float(np.float32(v)))


def export_tree_pipeline_to_c(artifact, out_dir):
    """Exact exporter (format version 3).  Writes esp32_dt_model.h with predict_esp32_dt(raw):
    no preprocessing arithmetic on the device; one float comparison per tree level."""
    pipe = artifact["pipeline"]
    le = artifact["label_encoder"]
    numeric_cols = artifact["numeric_cols"]
    tree = pipe.named_steps["clf"].tree_
    sel = pipe.named_steps["selector"].get_support(indices=True)
    median = pipe.named_steps["imputer"].statistics_.astype(np.float32)

    raw_feature, raw_thr = raw_space_thresholds(artifact)
    check_node_format(tree.node_count, len(numeric_cols), len(le.classes_))
    check = verify_raw_thresholds(artifact, raw_feature, raw_thr)
    check["edge_cases"] = verify_edge_cases(artifact, raw_feature, raw_thr)
    check["monotone"] = verify_monotone(artifact, raw_feature)
    check["exact"] = bool(check["exact"] and check["edge_cases"]["exact"] and check["monotone"]["exact"])
    if not check["exact"]:
        raise RuntimeError(f"raw-space threshold check failed: {check}")
    node_class = np.argmax(tree.value.reshape(tree.node_count, -1), axis=1)

    h = [
        "#pragma once",
        "// FaithTree exact export (v3): thresholds live in the raw float32 input space.",
        "// For every float32 input vector this function returns exactly what the fitted",
        "// scikit-learn pipeline returns (NaN/inf inputs are replaced by the training median).",
        "#include <stdint.h>",
        "#include <math.h>",
        "#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)",
        "#error \"FaithTree exact export needs IEEE 754 float semantics: build without -ffast-math / -ffinite-math-only\"",
        "#endif",
        f"#define N_RAW_FEATURES {len(numeric_cols)}",
        f"#define N_SELECTED_FEATURES {len(sel)}",
        f"#define N_NODES {tree.node_count}",
        f"#define N_CLASSES {len(le.classes_)}",
        f"#define FAITHTREE_EXPORT_VERSION 3",
        "static const float RAW_MEDIAN[N_RAW_FEATURES] = {" + ",".join(_c_hex(v) for v in median) + "};",
        "static const int16_t CHILD_LEFT[N_NODES] = {" + ",".join(str(int(v)) for v in tree.children_left) + "};",
        "static const int16_t CHILD_RIGHT[N_NODES] = {" + ",".join(str(int(v)) for v in tree.children_right) + "};",
        "static const uint8_t RAW_FEATURE[N_NODES] = {" + ",".join(str(int(v)) for v in raw_feature) + "};",
        "static const float RAW_THRESHOLD[N_NODES] = {" + ",".join(_c_hex(v) for v in raw_thr) + "};",
        "static const uint8_t NODE_CLASS[N_NODES] = {" + ",".join(str(int(v)) for v in node_class) + "};",
        "static const char* CLASS_NAMES[] = {" + ",".join(json.dumps(str(c)) for c in le.classes_) + "};",
        r'''
int predict_esp32_dt(const float raw[N_RAW_FEATURES]) {
  int node = 0;
  while (CHILD_LEFT[node] != -1) {
    int f = RAW_FEATURE[node];
    float x = raw[f];
    if (!isfinite(x)) x = RAW_MEDIAN[f];          /* same as inf->NaN->median in Python */
    node = (x <= RAW_THRESHOLD[node]) ? CHILD_LEFT[node] : CHILD_RIGHT[node];
  }
  return NODE_CLASS[node];
}
''',
    ]
    out_path = Path(out_dir) / "esp32_dt_model.h"
    out_path.write_text("\n".join(h), encoding="utf-8")
    (Path(out_dir) / "raw_threshold_check.json").write_text(json.dumps(check, indent=2))
    return out_path


def emulate_c_predict(artifact, X_raw):
    """numpy re-implementation of the exact predict_esp32_dt() (raw-space thresholds)."""
    pipe = artifact["pipeline"]
    tree = pipe.named_steps["clf"].tree_
    raw_feature, raw_thr = artifact.get("_raw_cache") or raw_space_thresholds(artifact)
    artifact["_raw_cache"] = (raw_feature, raw_thr)
    median = pipe.named_steps["imputer"].statistics_.astype(np.float32)
    raw = (X_raw[artifact["numeric_cols"]].to_numpy(dtype=np.float32) if hasattr(X_raw, "columns")
           else np.asarray(X_raw, dtype=np.float32))
    raw = np.where(np.isfinite(raw), raw, median[None, :]).astype(np.float32)
    left, right = tree.children_left, tree.children_right
    node_class = np.argmax(tree.value.reshape(tree.node_count, -1), axis=1)
    node = np.zeros(len(raw), dtype=np.int64)
    rows = np.arange(len(raw))
    active = left[node] != -1
    while active.any():
        a = rows[active]; n = node[a]
        go_left = raw[a, raw_feature[n]] <= raw_thr[n]
        node[a] = np.where(go_left, left[n], right[n])
        active = left[node] != -1
    return node_class[node]


def mismatch_report(artifact, X_raw, py_pred, c_pred, y_true=None, out_csv=None):
    """Rows where the C logic and scikit-learn disagree, with the deciding node on the sklearn path."""
    idx = np.where(np.asarray(py_pred) != np.asarray(c_pred))[0]
    rows = []
    if len(idx):
        pipe = artifact["pipeline"]; tree = pipe.named_steps["clf"].tree_
        Xs = X_raw.iloc[idx] if hasattr(X_raw, "iloc") else X_raw[idx]
        Z = np.asarray(pipe[:-1].transform(Xs)).astype(np.float32)
        for k, i in enumerate(idx):
            node, path = 0, []
            while tree.children_left[node] != -1:
                f, t = tree.feature[node], tree.threshold[node]
                path.append((node, int(f), float(Z[k, f]), float(t), float(np.float32(t))))
                node = tree.children_left[node] if Z[k, f] <= t else tree.children_right[node]
            close = min(path, key=lambda p: abs(p[2] - p[3]))
            rows.append(dict(row=int(i), true=None if y_true is None else int(np.asarray(y_true)[i]),
                             sklearn=int(np.asarray(py_pred)[i]), c=int(np.asarray(c_pred)[i]),
                             node=close[0], sel_feature=close[1], z_float32=repr(close[2]),
                             threshold_f64=repr(close[3]), threshold_rounded_f32=repr(close[4]),
                             gap=close[2] - close[3]))
    df = pd.DataFrame(rows)
    if out_csv is not None:
        df.to_csv(out_csv, index=False)
    return df


# -----------------------------------------------------------------------------
# Host compiler checks (Windows-safe: works when gcc lives in a folder with spaces)
# -----------------------------------------------------------------------------
def _find_cc():
    import os
    cc = None
    env_cc = os.environ.get("FAITHTREE_CC")
    if env_cc:
        env_cc = env_cc.strip().strip('"')
        if Path(env_cc).is_file():
            cc = env_cc
        else:
            print(f"[host C check] FAITHTREE_CC={env_cc!r} does not exist -- falling back to gcc on PATH")
    if cc is None:
        for d in ("C:\\w64devkit\\bin\\gcc.exe", "C:\\w64devkit\\w64devkit\\bin\\gcc.exe"):
            if os.name == "nt" and Path(d).is_file():
                cc = d
                break
    cc = cc or shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if cc and os.name == "nt" and " " in cc:
        try:                                    # 8.3 short path, e.g. C:\\PROGRA~1\\... (no spaces)
            import ctypes
            buf = ctypes.create_unicode_buffer(1024)
            if ctypes.windll.kernel32.GetShortPathNameW(cc, buf, 1024):
                cc = buf.value
        except Exception:
            pass
    return cc


def _run_cc(cc, args, cwd):
    import os
    env = dict(os.environ)
    if os.name == "nt":                          # let mingw find its own libs via a space-free PATH entry
        env["PATH"] = str(Path(cc).parent) + os.pathsep + env.get("PATH", "")
    try:
        return subprocess.run([cc] + args, capture_output=True, text=True, cwd=cwd, env=env)
    except OSError as e:                         # compiler path wrong / not executable
        return subprocess.CompletedProcess([cc] + args, 1, "", f"could not start compiler {cc!r}: {e}")


def host_c_check(esp32_dir, python_pred):
    """Compile esp32_dt_model.h + esp32_test_vectors.h with the host C compiler; compare with Python."""
    cc = _find_cc()
    if cc is None:
        return None
    esp32_dir = Path(esp32_dir)
    td = Path(tempfile.mkdtemp(prefix="ftcc_", dir=_safe_tmp()))
    (td / "main.c").write_text(HOST_MAIN)
    for f in ("esp32_dt_model.h", "esp32_test_vectors.h"):
        shutil.copy(esp32_dir / f, td / f)
    r = _run_cc(cc, ["-O2", "-o", "t.exe", "main.c", "-lm"], td)
    if r.returncode != 0:
        return {"compiled": False, "compiler": cc, "error": r.stderr[:800]}
    out = subprocess.run([str(td / "t.exe")], capture_output=True, text=True, cwd=td).stdout.split()
    shutil.rmtree(td, ignore_errors=True)
    c_pred = np.array([int(v) for v in out])
    py = np.asarray(python_pred)[: len(c_pred)]
    return {"compiled": True, "compiler": cc, "n": int(len(c_pred)), "matches": int((c_pred == py).sum()),
            "agreement": float((c_pred == py).mean())}


HOST_MAIN_FULL = r"""
#include <stdio.h>
#include <stdlib.h>
#include "esp32_dt_model.h"
int main(int argc, char** argv){
  FILE* fi = fopen(argv[1], "rb"); FILE* fo = fopen(argv[2], "wb");
  float row[N_RAW_FEATURES]; unsigned char p;
  while (fread(row, sizeof(float), N_RAW_FEATURES, fi) == N_RAW_FEATURES) {
    p = (unsigned char)predict_esp32_dt(row); fwrite(&p, 1, 1, fo);
  }
  fclose(fi); fclose(fo); return 0;
}
"""


def _safe_tmp():
    import os
    for d in (os.environ.get("FAITHTREE_TMP"), "C:\\fttmp" if os.name == "nt" else None, tempfile.gettempdir()):
        if d and " " not in d:
            Path(d).mkdir(parents=True, exist_ok=True)
            return d
    return tempfile.gettempdir()


def host_c_check_full(esp32_dir, artifact, X_raw, python_pred, chunk=500_000):
    """Compiled C header vs scikit-learn on EVERY row of X_raw (e.g. the whole test set)."""
    cc = _find_cc()
    if cc is None:
        return None
    td = Path(tempfile.mkdtemp(prefix="ftcc_", dir=_safe_tmp()))
    (td / "full.c").write_text(HOST_MAIN_FULL)
    shutil.copy(Path(esp32_dir) / "esp32_dt_model.h", td / "esp32_dt_model.h")
    r = _run_cc(cc, ["-O2", "-o", "full.exe", "full.c", "-lm"], td)
    if r.returncode != 0:
        return {"compiled": False, "compiler": cc, "error": r.stderr[:800]}
    raw = (X_raw[artifact["numeric_cols"]].to_numpy(dtype=np.float32) if hasattr(X_raw, "columns")
           else np.asarray(X_raw, dtype=np.float32))
    preds = []
    for s in range(0, len(raw), chunk):
        np.ascontiguousarray(raw[s:s + chunk]).tofile(td / "in.bin")
        subprocess.run([str(td / "full.exe"), "in.bin", "out.bin"], cwd=td, check=True)
        preds.append(np.fromfile(td / "out.bin", dtype=np.uint8))
    shutil.rmtree(td, ignore_errors=True)
    c_pred = np.concatenate(preds).astype(np.int64)
    py = np.asarray(python_pred)
    return {"compiled": True, "compiler": cc, "n": int(len(c_pred)), "matches": int((c_pred == py).sum()),
            "agreement": float((c_pred == py).mean()), "c_pred": c_pred}
