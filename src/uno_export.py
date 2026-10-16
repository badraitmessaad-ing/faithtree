"""
uno_export.py -- exact export for the Arduino Uno (ATmega328P, 32 KB flash, 2 KB RAM, no FPU).

Same raw-space thresholds as the ESP32 export (esp32_export.raw_space_thresholds), with three changes
for an 8-bit AVR board:
  * every array lives in flash (PROGMEM) and is read with pgm_read_*; 2 KB of RAM cannot hold the tree;
  * only the raw features that the tree actually uses are kept ("used features"), so each test vector
    is as short as possible;
  * the number of flashed test vectors is chosen so that model + vectors fit in the 32 KB flash.

The generated header also compiles on a PC (PROGMEM and pgm_read_* fall back to plain memory),
which is how host_check() verifies it.

  export_tree_pipeline_to_uno(artifact, out_dir)        -> uno_dt_model.h, used-feature list
  max_vectors_for_flash(n_nodes, n_used, ...)           -> how many test vectors fit
  create_uno_test_vectors_header(...)                   -> uno_test_vectors.h
  write_uno_sketch(out_dir)                             -> uno_dt_test.ino
  emulate_uno_predict(model_info, X_used)               -> numpy re-implementation of the C code
  host_check(out_dir, python_pred)                      -> compile with gcc on the PC and compare
"""
import json, os, subprocess, tempfile
from pathlib import Path
import numpy as np

import esp32_export as ex

UNO_FLASH_BYTES = 32_256          # 32 KB minus the 0.5 KB Optiboot bootloader
UNO_RESERVED_BYTES = 5_500        # Arduino core, Serial, the sketch itself (estimate; check the IDE's "Sketch uses")
BYTES_PER_NODE = 10               # int16 left + int16 right + uint8 feature + float32 threshold + uint8 class


def _c_float(v):
    v = float(np.float32(v))
    if np.isnan(v): return "NAN"
    if np.isinf(v): return "INFINITY" if v > 0 else "-INFINITY"
    s = repr(v)
    if "e" not in s and "." not in s: s += ".0"
    return s + "f"


def export_tree_pipeline_to_uno(artifact, out_dir):
    """Write uno_dt_model.h. Returns a dict describing the model (used features, medians, sizes)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pipe = artifact["pipeline"]; tree = pipe.named_steps["clf"].tree_
    numeric_cols = list(artifact["numeric_cols"])
    raw_feature, raw_thr = artifact.get("_raw_cache") or ex.raw_space_thresholds(artifact)
    ex.check_node_format(tree.node_count, len(numeric_cols), len(artifact["label_encoder"].classes_))
    check = ex.verify_raw_thresholds(artifact, raw_feature, raw_thr)
    check["edge_cases"] = ex.verify_edge_cases(artifact, raw_feature, raw_thr)
    check["exact"] = bool(check["exact"] and check["edge_cases"]["exact"])
    if not check["exact"]:
        raise RuntimeError(f"raw-space threshold check failed: {check}")

    internal = tree.children_left != -1
    used = sorted(set(int(f) for f in raw_feature[internal]))
    if len(used) > 255:
        raise ValueError("more than 255 used features")
    compact = {f: i for i, f in enumerate(used)}
    feat_c = np.array([compact[int(f)] if internal[n] else 0 for n, f in enumerate(raw_feature)], dtype=np.int64)
    median_all = pipe.named_steps["imputer"].statistics_.astype(np.float32)
    median_used = median_all[used]
    node_class = np.argmax(tree.value.reshape(tree.node_count, -1), axis=1)

    def arr(ctype, name, vals, fmt=str):
        return f"static const {ctype} {name}[{len(vals)}] PROGMEM = {{" + ",".join(fmt(v) for v in vals) + "};"

    h = [
        "// FaithTree exact export for the Arduino Uno (ATmega328P).",
        "// Thresholds are in the raw float32 input space: the board compares raw values only.",
        "#pragma once",
        "#include <stdint.h>",
        "#include <math.h>",
        "#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)",
        "#error \"FaithTree exact export needs IEEE 754 float semantics: build without -ffast-math / -ffinite-math-only\"",
        "#endif",
        "#ifdef __AVR__",
        "#include <avr/pgmspace.h>",
        "#else",
        "#define PROGMEM",
        "#define pgm_read_byte(p)  (*(const uint8_t *)(p))",
        "#define pgm_read_word(p)  (*(const uint16_t *)(p))",
        "#define pgm_read_dword(p) (*(const uint32_t *)(p))",
        "#define pgm_read_float(p) (*(const float *)(p))",
        "#endif",
        "",
        "#define FAITHTREE_EXPORT_VERSION 3",
        f"#define N_NODES {tree.node_count}",
        f"#define N_USED_FEATURES {len(used)}",
        f"#define N_CLASSES {int(tree.value.shape[-1])}",
        "",
        "// position of each used feature in the model's numeric_cols list:",
        "// " + ", ".join(f"{i}={numeric_cols[f]}" for i, f in enumerate(used)),
        arr("float", "USED_MEDIAN", median_used, _c_float),
        arr("int16_t", "CHILD_LEFT", tree.children_left.astype(int)),
        arr("int16_t", "CHILD_RIGHT", tree.children_right.astype(int)),
        arr("uint8_t", "NODE_FEATURE", feat_c),
        arr("float", "NODE_THRESHOLD", raw_thr, _c_float),
        arr("uint8_t", "NODE_CLASS", node_class),
        "",
        "// x = the N_USED_FEATURES raw values of one flow, in the order listed above.",
        "static inline uint8_t predict_uno_dt(const float *x) {",
        "  int16_t node = 0;",
        "  int16_t left = (int16_t)pgm_read_word(&CHILD_LEFT[0]);",
        "  while (left != -1) {",
        "    uint8_t f = pgm_read_byte(&NODE_FEATURE[node]);",
        "    float v = x[f];",
        "    if (!isfinite(v)) v = pgm_read_float(&USED_MEDIAN[f]);",
        "    node = (v <= pgm_read_float(&NODE_THRESHOLD[node])) ? left : (int16_t)pgm_read_word(&CHILD_RIGHT[node]);",
        "    left = (int16_t)pgm_read_word(&CHILD_LEFT[node]);",
        "  }",
        "  return pgm_read_byte(&NODE_CLASS[node]);",
        "}",
    ]
    path = out_dir / "uno_dt_model.h"
    path.write_text("\n".join(h) + "\n", encoding="utf-8")
    info = {"n_nodes": int(tree.node_count), "n_used_features": len(used), "used_raw_index": used,
            "used_columns": [numeric_cols[f] for f in used], "model_flash_bytes": int(tree.node_count * BYTES_PER_NODE + 4 * len(used)),
            "raw_threshold_check": check, "header": str(path)}
    (out_dir / "uno_model_info.json").write_text(json.dumps(info, indent=2, default=str))
    info["_arrays"] = (tree.children_left, tree.children_right, feat_c, raw_thr.astype(np.float32), node_class, median_used)
    return info


def max_vectors_for_flash(n_nodes, n_used, flash=UNO_FLASH_BYTES, reserved=UNO_RESERVED_BYTES):
    """How many test vectors fit after the model and the Arduino core."""
    model = n_nodes * BYTES_PER_NODE + 4 * n_used
    per_vector = 4 * n_used + 1 + 1 + 4                     # features + true label + python pred + sample id
    room = flash - reserved - model
    return max(0, room // per_vector), model


def create_uno_test_vectors_header(X_used, y_true, y_pred, sample_ids, out_dir):
    X_used = np.asarray(X_used, dtype=np.float32)
    n, d = X_used.shape
    lines = ["#pragma once", "#include <stdint.h>", "#include <math.h>", "",
             f"#define TEST_SAMPLE_COUNT {n}", "",
             "static const uint32_t TEST_SAMPLE_ID[TEST_SAMPLE_COUNT] PROGMEM = {" + ",".join(str(int(i)) for i in sample_ids) + "};",
             "static const uint8_t TEST_TRUE_LABEL[TEST_SAMPLE_COUNT] PROGMEM = {" + ",".join(str(int(v)) for v in y_true) + "};",
             "static const uint8_t TEST_PYTHON_PRED[TEST_SAMPLE_COUNT] PROGMEM = {" + ",".join(str(int(v)) for v in y_pred) + "};",
             f"static const float TEST_X[TEST_SAMPLE_COUNT][{d}] PROGMEM = {{"]
    for row in X_used:
        lines.append("  {" + ",".join(_c_float(v) for v in row) + "},")
    lines.append("};")
    path = Path(out_dir) / "uno_test_vectors.h"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


UNO_SKETCH = r"""
// FaithTree benchmark for the Arduino Uno (ATmega328P @ 16 MHz).
// Times ONLY predict_uno_dt(); every result is printed after its timing ends.
// Board: Arduino Uno. Serial Monitor: 115200 baud. Wait for "#DONE", then copy the whole output.
#include <avr/pgmspace.h>
#include "uno_dt_model.h"
#include "uno_test_vectors.h"

#define REPS 16                     // repeated calls per sample: micros() has a 4 us step on the Uno

float x[N_USED_FEATURES];
volatile uint8_t g_sink;
extern int __heap_start, *__brkval;

int freeRam() { int v; return (int)&v - (__brkval == 0 ? (int)&__heap_start : (int)__brkval); }

void loadRow(uint16_t i) {
  for (uint8_t f = 0; f < N_USED_FEATURES; f++) x[f] = pgm_read_float(&TEST_X[i][f]);
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println(F("#META,begin"));
  Serial.println(F("#META,board,Arduino Uno (ATmega328P)"));
  Serial.print(F("#META,cpu_mhz,"));          Serial.println(F_CPU / 1000000UL);
  Serial.print(F("#META,n_nodes,"));          Serial.println(N_NODES);
  Serial.print(F("#META,n_used_features,"));  Serial.println(N_USED_FEATURES);
  Serial.print(F("#META,n_samples,"));        Serial.println(TEST_SAMPLE_COUNT);
  Serial.print(F("#META,reps_per_sample,"));  Serial.println(REPS);
  Serial.print(F("#META,free_ram_bytes,"));   Serial.println(freeRam());
  Serial.println(F("#META,end"));
  Serial.println(F("sample_id,true_label,python_pred,board_pred,match,infer_us"));

  uint16_t matches = 0;
  float sum_us = 0, max_us = 0;
  for (uint16_t i = 0; i < TEST_SAMPLE_COUNT; i++) {
    loadRow(i);                                    // copy the vector from flash to RAM (not timed)
    uint8_t p = predict_uno_dt(x);
    uint32_t t0 = micros();
    for (uint8_t r = 0; r < REPS; r++) g_sink = predict_uno_dt(x);
    uint32_t dt = micros() - t0;
    float us = dt / (float)REPS;
    uint8_t py = pgm_read_byte(&TEST_PYTHON_PRED[i]);
    uint8_t ok = (p == py);
    matches += ok; sum_us += us; if (us > max_us) max_us = us;
    Serial.print(pgm_read_dword(&TEST_SAMPLE_ID[i])); Serial.print(',');
    Serial.print(pgm_read_byte(&TEST_TRUE_LABEL[i])); Serial.print(',');
    Serial.print(py); Serial.print(','); Serial.print(p); Serial.print(',');
    Serial.print(ok); Serial.print(','); Serial.println(us, 2);
  }
  Serial.print(F("SUMMARY_MATCHES,")); Serial.print(matches); Serial.print('/'); Serial.println(TEST_SAMPLE_COUNT);
  Serial.print(F("INFER_US_MEAN,"));   Serial.println(sum_us / TEST_SAMPLE_COUNT, 2);
  Serial.print(F("INFER_US_MAX,"));    Serial.println(max_us, 2);
  Serial.println(F("#DONE"));
}

void loop() {}
"""


def write_uno_sketch(out_dir):
    path = Path(out_dir) / (Path(out_dir).name + ".ino")      # the Arduino IDE wants folder name == sketch name
    path.write_text(UNO_SKETCH.lstrip(), encoding="utf-8")
    return path


def emulate_uno_predict(info, X_used):
    """numpy re-implementation of predict_uno_dt() (float32 compare, median for non-finite values)."""
    left, right, feat, thr, cls, med = info["_arrays"]
    X = np.asarray(X_used, dtype=np.float32).copy()
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.broadcast_to(med, X.shape)[bad]
    node = np.zeros(len(X), dtype=np.int64)
    rows = np.arange(len(X))
    active = left[node] != -1
    while active.any():
        r = rows[active]; n = node[r]
        go_left = X[r, feat[n]] <= thr[n]
        node[r] = np.where(go_left, left[n], right[n])
        active = left[node] != -1
    return cls[node]


HOST_MAIN = r"""
#include <stdio.h>
#include "uno_dt_model.h"
#include "uno_test_vectors.h"
int main(void) {
  float x[N_USED_FEATURES];
  for (int i = 0; i < TEST_SAMPLE_COUNT; i++) {
    for (int f = 0; f < N_USED_FEATURES; f++) x[f] = TEST_X[i][f];
    printf("%d\n", predict_uno_dt(x));
  }
  return 0;
}
"""


def host_check(out_dir, python_pred):
    """Compile the Uno header with the PC's C compiler (no AVR needed) and compare with Python."""
    cc = ex._find_cc()
    if cc is None:
        return None
    out_dir = Path(out_dir)
    with tempfile.TemporaryDirectory(dir=ex._safe_tmp()) as td:
        td = Path(td)
        (td / "main.c").write_text(HOST_MAIN)
        for f in ("uno_dt_model.h", "uno_test_vectors.h"):
            (td / f).write_bytes((out_dir / f).read_bytes())
        exe = td / ("t.exe" if os.name == "nt" else "t")
        r = ex._run_cc(cc, ["-O2", "-std=c99", "-o", str(exe), "main.c", "-lm"], td)
        if r is None or r.returncode != 0:
            return {"compiled": False, "stderr": (r.stderr[-800:] if r is not None else "compiler not runnable")}
        out = subprocess.run([str(exe)], capture_output=True, text=True, cwd=td).stdout.split()
    pred = np.array([int(v) for v in out])
    py = np.asarray(python_pred)
    return {"compiled": True, "compiler": cc, "n": int(len(py)), "matches": int((pred == py).sum()),
            "agreement": float((pred == py).mean())}
