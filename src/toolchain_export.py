"""
toolchain_export.py -- on-device comparison helpers (notebook 11).

Goal: compare the framework's exporter with EXISTING toolchains (emlearn, micromlgen) and with other model families
(random forest, a shallow depth-5 tree like the one used in prior ESP32 IDS work) under IDENTICAL conditions:
  * same preprocessing (impute -> scale -> select, in float32, generated from the same arrays),
  * same classifier input (the selected, scaled features),
  * same test vectors and the same benchmark sketch (it only calls predict_esp32_dt()).
Only the *classifier code* differs between variants.

Contents
  export_forest_to_c                  our own forest exporter (hard majority vote, same API as the tree exporter)
  export_external_classifier_to_c     preprocessing + any external classifier code (emlearn / micromlgen / ...)
  emlearn_code / micromlgen_code      generate competitor code from a fitted sklearn tree / forest
  forest_hardvote_predict             Python reference for a forest (majority vote of per-tree leaf classes)
  emulate_c_forest_predict            float32 emulation of the forest C code (whole test split, no compiler needed)
  host_cpp_check / host_footprint     compile the generated header with g++ and (a) compare predictions, (b) measure code+rodata bytes
"""
import json, shutil, subprocess, tempfile, re
from pathlib import Path
import numpy as np

PREPROC_FN = r'''
static inline float safe_value(float x) {
  if (!isfinite(x)) return 0.0f;
  if (x > 1e10f) return 1e10f;
  if (x < -1e10f) return -1e10f;
  return x;
}

static inline void esp32_preprocess(const float raw[N_RAW_FEATURES], float selected[N_SELECTED_FEATURES]) {
  for (int j = 0; j < N_SELECTED_FEATURES; ++j) {
    int i = SELECTED_IDX[j];
    float x = raw[i];
    if (!isfinite(x)) { x = IMPUTER_MEDIAN_SELECTED[j]; }
    x = safe_value(x);
    x = (x - SCALER_CENTER_SELECTED[j]) / SCALER_SCALE_SELECTED[j];
    x = safe_value(x);
    selected[j] = x;
  }
}
'''


def _fmt(vals):
    return ",".join("%.9g" % float(v) for v in vals)


def _preproc_header(artifact, n_nodes, extra_defines=()):
    pipe, le, cols = artifact["pipeline"], artifact["label_encoder"], artifact["numeric_cols"]
    imputer, scaler, selector = pipe.named_steps["imputer"], pipe.named_steps["scaler"], pipe.named_steps["selector"]
    sel = selector.get_support(indices=True)
    h = ["#pragma once", "#include <stdint.h>", "#include <math.h>",
         f"#define N_RAW_FEATURES {len(cols)}", f"#define N_SELECTED_FEATURES {len(sel)}",
         f"#define N_NODES {int(n_nodes)}", f"#define N_CLASSES {len(le.classes_)}"] + list(extra_defines)
    h.append("static const float IMPUTER_MEDIAN_SELECTED[] = {" + _fmt(imputer.statistics_[sel]) + "};")
    h.append("static const float SCALER_CENTER_SELECTED[] = {" + _fmt(scaler.center_[sel]) + "};")
    h.append("static const float SCALER_SCALE_SELECTED[] = {" + _fmt(scaler.scale_[sel]) + "};")
    h.append("static const uint8_t SELECTED_IDX[] = {" + ",".join(str(int(v)) for v in sel) + "};")
    h.append(PREPROC_FN)
    return h


# ----------------------------------------------------------------------------- our forest exporter
def export_forest_to_c(artifact, out_dir):
    """Random forest -> C. Hard majority vote of per-tree leaf classes (ties -> lowest class index, like np.argmax).
    Per-tree node indices are LOCAL (int16), TREE_OFFSET gives the start of each tree: 11 bytes per node, like the single-tree export."""
    clf = artifact["pipeline"].named_steps["clf"]
    trees = [e.tree_ for e in clf.estimators_]
    counts = [t.node_count for t in trees]
    assert max(counts) < 32768, "a tree has more than 32767 nodes: local int16 indices are not enough"
    offs = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    feat = np.concatenate([t.feature for t in trees]); left = np.concatenate([t.children_left for t in trees])
    right = np.concatenate([t.children_right for t in trees]); thr = np.concatenate([t.threshold for t in trees])
    cls = np.concatenate([np.argmax(t.value.reshape(t.node_count, -1), axis=1) for t in trees])
    h = _preproc_header(artifact, int(offs[-1]), [f"#define N_TREES {len(trees)}"])
    h.append("static const int16_t FEATURE[] = {" + ",".join(str(int(v)) for v in feat) + "};")
    h.append("static const int16_t CHILD_LEFT[] = {" + ",".join(str(int(v)) for v in left) + "};")
    h.append("static const int16_t CHILD_RIGHT[] = {" + ",".join(str(int(v)) for v in right) + "};")
    h.append("static const float THRESHOLD[] = {" + _fmt(thr) + "};")
    h.append("static const uint8_t NODE_CLASS[] = {" + ",".join(str(int(v)) for v in cls) + "};")
    h.append("static const int32_t TREE_OFFSET[] = {" + ",".join(str(int(v)) for v in offs) + "};")
    h.append(r'''
int predict_esp32_dt(const float raw[N_RAW_FEATURES]) {
  float selected[N_SELECTED_FEATURES];
  esp32_preprocess(raw, selected);
  uint16_t votes[N_CLASSES];
  for (int c = 0; c < N_CLASSES; ++c) votes[c] = 0;
  for (int t = 0; t < N_TREES; ++t) {
    const int base = TREE_OFFSET[t];
    int node = 0;
    while (CHILD_LEFT[base + node] != -1) {
      const int f = FEATURE[base + node];
      node = (selected[f] <= THRESHOLD[base + node]) ? CHILD_LEFT[base + node] : CHILD_RIGHT[base + node];
    }
    votes[NODE_CLASS[base + node]]++;
  }
  int best = 0;
  for (int c = 1; c < N_CLASSES; ++c) if (votes[c] > votes[best]) best = c;
  return best;
}
''')
    p = Path(out_dir) / "esp32_dt_model.h"; p.write_text("\n".join(h), encoding="utf-8"); return p


# ----------------------------------------------------------------------------- competitor code
def import_emlearn_safely():
    """Import emlearn WITHOUT letting it change numpy's global error handling.
    emlearn (emlearn/bayes.py) calls numpy.seterr(all='raise') at import time. After that, any float32 overflow while pandas reads the CSV
    (the NetFlow files contain counters outside the float32 range) raises FloatingPointError instead of becoming inf. We restore numpy's state."""
    old = np.geterr()
    try:
        import emlearn
        return emlearn
    finally:
        np.seterr(**old)


def emlearn_code(clf, name="emlc"):
    """emlearn 'inline' C code with FLOAT features (its default int16 features would truncate the scaled inputs)."""
    old = np.geterr()
    try:
        emlearn = import_emlearn_safely()
        m = emlearn.convert(clf, method="inline", dtype="float")
        code = m.save(name=name, file=None, inference=["inline"])
    finally:
        np.seterr(**old)                       # emlearn may re-apply seterr(all='raise') when it lazily imports submodules
    return code, f"{name}_predict(selected, N_SELECTED_FEATURES)", ""


def micromlgen_code(clf):
    from micromlgen import port
    code = port(clf)
    cls = "RandomForest" if hasattr(clf, "estimators_") else "DecisionTree"
    return code, f"g_clf.predict(selected)", f"static Eloquent::ML::Port::{cls} g_clf;"


def export_external_classifier_to_c(artifact, out_dir, classifier_code, call_expr, n_nodes, declaration=""):
    """Our preprocessing + an external classifier. Result header exposes the SAME predict_esp32_dt() as the framework's own export."""
    code = "\n".join(l for l in classifier_code.splitlines() if l.strip() != "#pragma once")
    h = _preproc_header(artifact, n_nodes)
    h.append(code)
    if declaration: h.append(declaration)
    h.append(f"int predict_esp32_dt(const float raw[N_RAW_FEATURES]) {{\n  float selected[N_SELECTED_FEATURES];\n  esp32_preprocess(raw, selected);\n  return {call_expr};\n}}\n")
    p = Path(out_dir) / "esp32_dt_model.h"; p.write_text("\n".join(h), encoding="utf-8"); return p


# ----------------------------------------------------------------------------- Python references / emulation
def forest_hardvote_predict(pipe, X):
    """Reference for the forest C code: preprocessing of the fitted pipeline, then majority vote of the per-tree leaf classes (ties -> lowest index)."""
    Xs = X
    for _, step in pipe.steps[:-1]:          # apply the fitted preprocessing steps one by one (a sliced Pipeline refuses to transform: stateless last step)
        Xs = step.transform(Xs)
    clf = pipe.named_steps["clf"]
    n_classes = clf.n_classes_; votes = np.zeros((len(Xs), n_classes), dtype=np.int32)
    for e in clf.estimators_:
        t = e.tree_; leaf_class = np.argmax(t.value.reshape(t.node_count, -1), axis=1)
        votes[np.arange(len(Xs)), leaf_class[e.apply(np.asarray(Xs, dtype=np.float32))]] += 1
    return votes.argmax(axis=1)


def _selected_f32(artifact, X_raw):
    pipe, cols = artifact["pipeline"], artifact["numeric_cols"]
    imp, sc, selr = pipe.named_steps["imputer"], pipe.named_steps["scaler"], pipe.named_steps["selector"]
    sel = selr.get_support(indices=True)
    raw = X_raw[cols].to_numpy(dtype=np.float32) if hasattr(X_raw, "columns") else np.asarray(X_raw, dtype=np.float32)
    x = raw[:, sel].copy(); med = imp.statistics_[sel].astype(np.float32)
    ctr, scl = sc.center_[sel].astype(np.float32), sc.scale_[sel].astype(np.float32)
    x = np.where(~np.isfinite(x), np.broadcast_to(med, x.shape), x).astype(np.float32)
    safe = lambda v: np.clip(np.where(np.isfinite(v), v, np.float32(0)), np.float32(-1e10), np.float32(1e10)).astype(np.float32)
    return safe(((safe(x) - ctr) / scl).astype(np.float32))


def emulate_c_forest_predict(artifact, X_raw):
    """float32 emulation of export_forest_to_c() on any number of rows (no compiler needed)."""
    x = _selected_f32(artifact, X_raw); clf = artifact["pipeline"].named_steps["clf"]
    votes = np.zeros((len(x), clf.n_classes_), dtype=np.int32); rows = np.arange(len(x))
    for e in clf.estimators_:
        t = e.tree_; left, right, feat = t.children_left, t.children_right, t.feature
        thr = t.threshold.astype(np.float32); leaf_class = np.argmax(t.value.reshape(t.node_count, -1), axis=1)
        node = np.zeros(len(x), dtype=np.int64); act = left[node] != -1
        while act.any():
            a = rows[act]; n = node[a]
            node[a] = np.where(x[a, feat[n]] <= thr[n], left[n], right[n]); act = left[node] != -1
        votes[rows, leaf_class[node]] += 1
    return votes.argmax(axis=1)


# ----------------------------------------------------------------------------- host checks (g++)
HOST_MAIN_CPP = r"""
#include <stdio.h>
#include "esp32_dt_model.h"
#include "esp32_test_vectors.h"
int main(){ for (int i = 0; i < TEST_SAMPLE_COUNT; i++) printf("%d\n", predict_esp32_dt(TEST_X[i])); return 0; }
"""
FOOTPRINT_CPP = r"""
#include "esp32_dt_model.h"
volatile float g_in[N_RAW_FEATURES];   /* volatile input: stops the compiler folding the whole tree away */
volatile int g_sink;
int main(){ float x[N_RAW_FEATURES]; for (int i = 0; i < N_RAW_FEATURES; i++) x[i] = g_in[i];
            g_sink = predict_esp32_dt(x); return g_sink; }
"""


def _cxx():
    """C++ compiler; on Windows uses the 8.3 short path so a gcc in a folder whose path contains spaces works."""
    import os
    cxx = None
    cc = os.environ.get("FAITHTREE_CC", "").strip().strip('"')
    if cc and Path(cc).is_file():                       # g++ next to the configured gcc
        for name in ("g++.exe", "g++"):
            if (Path(cc).parent / name).is_file():
                cxx = str(Path(cc).parent / name); break
    cxx = cxx or shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if cxx and os.name == "nt" and " " in cxx:
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(1024)
            if ctypes.windll.kernel32.GetShortPathNameW(cxx, buf, 1024):
                cxx = buf.value
        except Exception:
            pass
    return cxx


def _env_for(cxx):
    import os
    env = dict(os.environ)
    env["PATH"] = str(Path(cxx).parent) + os.pathsep + env.get("PATH", "")
    return env


def _tmpdir():
    import os
    for d in (os.environ.get("FAITHTREE_TMP"), "C:\\fttmp" if os.name == "nt" else None):
        if d and " " not in d:
            Path(d).mkdir(parents=True, exist_ok=True)
            return tempfile.TemporaryDirectory(dir=d)
    return tempfile.TemporaryDirectory()


def _size_tool(cxx):
    for name in ("size.exe", "size"):
        p = Path(cxx).parent / name
        if p.is_file():
            return str(p)
    return shutil.which("size")


def host_cpp_check(variant_dir, python_pred, timeout=900):
    """Compile model + vectors with the host C++ compiler, run on all vectors, compare with python_pred."""
    cxx = _cxx()
    if cxx is None: return None
    d = Path(variant_dir)
    with _tmpdir() as td:
        td = Path(td); (td / "main.cpp").write_text(HOST_MAIN_CPP)
        for f in ("esp32_dt_model.h", "esp32_test_vectors.h"): shutil.copy(d / f, td / f)
        try:
            r = subprocess.run([cxx, "-O1", "-std=c++11", "-o", "t.exe", "main.cpp", "-lm"], capture_output=True, text=True,
                               timeout=timeout, cwd=td, env=_env_for(cxx))
        except OSError as e:
            return {"compiled": False, "error": f"could not start {cxx!r}: {e}"}
        if r.returncode != 0: return {"compiled": False, "error": r.stderr[:600]}
        out = subprocess.run([str(td / "t.exe")], capture_output=True, text=True, cwd=td).stdout.split()
    c = np.array([int(v) for v in out]); py = np.asarray(python_pred)[: len(c)]
    return {"compiled": True, "n": int(len(c)), "matches": int((c == py).sum()), "agreement": float((c == py).mean())}


def host_footprint(variant_dir, timeout=900):
    """Bytes of code + read-only data of the model alone (x86-64, -Os, gc-sections): a PROXY for ESP32 flash, useful for RELATIVE comparison only."""
    cxx = _cxx()
    size = _size_tool(cxx) if cxx else None
    if cxx is None or size is None: return None
    d = Path(variant_dir)
    with _tmpdir() as td:
        td = Path(td); (td / "main.cpp").write_text(FOOTPRINT_CPP); shutil.copy(d / "esp32_dt_model.h", td / "esp32_dt_model.h")
        try:
            r = subprocess.run([cxx, "-Os", "-std=c++11", "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections", "-o", "t.exe", "main.cpp", "-lm"],
                               capture_output=True, text=True, timeout=timeout, cwd=td, env=_env_for(cxx))
        except OSError:
            return None
        if r.returncode != 0: return None
        base = subprocess.run([cxx, "-Os", "-std=c++11", "-o", "b.exe", "-x", "c++", "-", "-lm"], input="#include <math.h>\nint main(){return 0;}",
                              capture_output=True, text=True, cwd=td, env=_env_for(cxx))
        def tot(exe):
            out = subprocess.run([size, "-A", str(exe)], capture_output=True, text=True).stdout
            return sum(int(m.group(2)) for m in re.finditer(r"^(\.text\S*|\.rodata\S*|\.rdata\S*)\s+(\d+)", out, re.M))
        return max(tot(td / "t.exe") - (tot(td / "b.exe") if base.returncode == 0 else 0), 0)
