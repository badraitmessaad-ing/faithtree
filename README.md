# FaithTree

**Exact and leakage-aware deployment of decision-tree intrusion detection on ESP32 microcontrollers**

When a scikit-learn tree is exported to C for a microcontroller, the preprocessing (imputation, clipping, scaling) is usually rewritten
in float32 on the device and the thresholds are rounded to decimal literals. Most of the time the board gives the same answer as
Python. Sometimes it doesn't, and nobody checks.

FaithTree avoids the problem. Every threshold of the tree is moved back into the raw float32 input space, so the device compares raw
feature values with stored constants and does no preprocessing arithmetic at all. We prove that the exported code returns the
scikit-learn prediction for **every** float32 input (NaN, ±inf and ±0 included) under assumptions that the exporter checks before it
writes a header. The repository also contains the leakage-aware evaluation protocol around it: validation-only selection, unseen-host,
temporal and cross-dataset tests, and a fresh-split replication of the whole model selection.

This is the code, the notebooks, the board logs and the result files behind the paper:

> B. Ait Messaad and S. Motahhir, "FaithTree: Exact and leakage-aware deployment of decision-tree intrusion detection on ESP32 microcontrollers,"
> submitted to *IEEE Internet of Things Journal*, 2026.

## Results at a glance (NF-ToN-IoT-v2, ten classes)

| | |
|---|---|
| Test MCC / macro-F1 | 0.9722 / 0.9091, with 7,073 nodes and depth 20 |
| Same selection on three new splits | test MCC 0.9721 ± 0.0011 ([evidence](evidence/fresh_split_replication/nf_ton_iot_v2)) |
| Unconstrained tree, same rows | 0.9727 MCC, but 614,847 nodes |
| Model size on the device | 70,730 B (10 B per node) |
| ESP32-D0WD-V3 at 240 MHz | 12.0 µs per flow on average, p99 32 µs (classifier only); 5,000/5,000 flows agree with scikit-learn |
| Compiled C on the host | 3,388,100/3,388,100 test rows agree |
| Adversarial inputs (3 × 100,000) | exact export 0 mismatches; nearest-float32 export 1,270 and 1,065 ([evidence](evidence/exact_export_verification)) |
| emlearn 0.23.2, same board and tree | 1,922/2,000 flows agree, 27.7 µs |

The latencies cover classification of a feature vector that is already computed. Packet capture, flow tracking and feature extraction
are not part of the measurement.

## How the exact export works

Between a raw feature and the comparison in a tree node, the pipeline does: ±inf → missing, median imputation, clipping,
`RobustScaler`, a float32 cast and feature selection. Each of these steps is monotone non-decreasing in the raw value, so for every
internal node there is a largest float32 raw value `R` that still goes left. We find `R` by bisection over the ordered float32 bit
patterns, evaluating the fitted pipeline itself (at most 32 evaluations per node), and the device only has to run this:

```c
int predict_esp32_dt(const float raw[N_RAW_FEATURES]) {
  int node = 0;
  while (CHILD_LEFT[node] != -1) {
    int f = RAW_FEATURE[node];
    float x = raw[f];
    if (!isfinite(x)) x = RAW_MEDIAN[f];        /* same as inf -> NaN -> median in Python */
    node = (x <= RAW_THRESHOLD[node]) ? CHILD_LEFT[node] : CHILD_RIGHT[node];
  }
  return NODE_CLASS[node];
}
```

The proof (Proposition 1 in the paper) needs four things, and `src/esp32_export.py` checks each of them before writing a header:

- the cut at every node is right: `R` goes left and the next float32 goes right (`verify_raw_thresholds`);
- NaN, ±inf, −0, +0 and ±FLT_MAX are routed the same way by the device rule and by the pipeline at every node, and every imputer
  median is a float32 number (`verify_edge_cases`);
- every used feature is monotone over 2,049 points spread across the float32 range (`verify_monotone`);
- the tree fits the 10-byte node: at most 32,767 nodes (`int16_t` child indices), 256 raw features and 256 classes (`check_node_format`).

If any check fails, nothing is written. One thing the exporter cannot check for you: build the firmware without fast-math options,
because `-ffast-math` lets the compiler drop the `isfinite()` test.

## Repository layout

```
faithtree/
├── run_all.py             runs the notebooks in order
├── requirements.txt
├── configs/               the fixed hyper-parameters of the paper's model
├── src/                   pipeline, search, export, evaluation helpers
├── notebooks/             one notebook per step (table below)
├── tools/                 export checks, fresh-split replication, Arduino Uno export, serial capture, synthetic data
├── esp32/serial_logs/     Serial Monitor logs of every board run in the paper
├── evidence/              small result files behind specific claims (see evidence/README.md)
├── figures/               script that redraws every figure of the paper (see figures/README.md)
├── data/                  put the CSV files here (not tracked)
├── results/               generated: models, metrics, headers (not tracked)
└── paper/                 generated: figures and tables (not tracked)
```

| Notebook | What it does |
|---|---|
| 01 | Original three-stage search (Silverback → Dolphin → pruning), selection on validation |
| 08 | Trains the paper's fixed configuration; K chosen on validation (used for the reported model) |
| 02 | Per-class metrics, baselines on the same split, ablations, class weights |
| 03 | ESP32 export (exact and legacy), test vectors, benchmark sketch, host checks |
| 04 | Tuning methods compared at an equal evaluation budget |
| 05 | Board logs: agreement, latency, flash, heap |
| 06 | Paper figures and tables |
| 07 | Feature-policy ablation (ports, identifier name patterns) |
| 09 | Binary (benign vs. attack) results |
| 10 | Ceiling analysis: ensembles and derived features |
| 11 | Exporter and model-family comparison (emlearn, micromlgen, forests) |
| 12 | Unseen-host and time-based splits |
| 14 | Cross-dataset test (NF-ToN-IoT-v2 ↔ NF-UNSW-NB15-v3) |
| 15 | Board logs of the exporter comparison |

| Tool | What it does |
|---|---|
| `verify_exact_export.py` | all export checks on a saved model, plus 3 × 100,000 adversarial inputs, without the dataset |
| `fresh_split_replication.py` | reruns the whole model selection on new splits and scores each test split once |
| `export_uno.py`, `analyze_uno_log.py` | Arduino Uno version of the export and analysis of its log |
| `capture_serial.py` | saves the board output to `esp32/serial_logs/` |
| `diagnose_exporter_mismatch.py` | re-traces the flashed vectors to show why emlearn disagrees with scikit-learn on the board |
| `make_synthetic_data.py` | small fake datasets for a dry run |

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/badraitmessaad-ing/faithtree.git
cd faithtree
pip install -r requirements.txt
```

The paper used scikit-learn 1.6.1. A saved model (`.pkl`) only loads with the scikit-learn version that created it.
A C compiler (gcc or clang) is optional; with one, notebooks 03 and 11 and `verify_exact_export.py` also compile and run the
generated header. On Windows, if the compiler path contains a space, set `FAITHTREE_CC` to the full path of `gcc.exe`.

## Data

Put the CSV files in `data/` with these names (or point `PAPER_DATA_DIR`, or `NF_TON_IOT_V2_CSV`, `CIC_TON_IOT_CSV`,
`NF_UNSW_NB15_V3_CSV`, to them):

| File | Dataset |
|---|---|
| `NF-ToN-IoT-v2.csv` | NF-ToN-IoT-v2 (NetFlow v2 features) |
| `CIC-ToN-IoT.csv` | CIC-ToN-IoT (CICFlowMeter features) |
| `NF-UNSW-NB15-v3.csv` | NF-UNSW-NB15-v3 (NetFlow v3 features) |

The datasets are published by the University of Queensland (https://staff.itee.uq.edu.au/marius/NIDS_datasets/) and are not
redistributed here. Please cite the original dataset papers if you use them.

## Try it without the real data

```bash
python tools/make_synthetic_data.py --out data_synthetic
export PAPER_DATA_DIR=data_synthetic          # Windows: set PAPER_DATA_DIR=data_synthetic
python run_all.py --mode smoke --datasets nf_unsw_nb15_v3 --steps 8 2 3
```

The numbers mean nothing on synthetic data; the run only shows that the pipeline works end to end.

## Reproducing the paper

`--mode` (or `PAPER_MODE`) sets every sample size: `smoke` takes minutes, `quick` is about ten times cheaper than the paper, and
`full` uses the paper settings (hours on the large datasets). All main splits use seed 42.

```bash
python run_all.py --check                                         # data and packages
python run_all.py --mode full --steps 8 2 3                       # model, evaluation, ESP32 export (each dataset)
python run_all.py --mode full --steps 4 7 10 11 12 --datasets nf_ton_iot_v2
python run_all.py --mode full --binary --steps 8 2                # binary profiles
python run_all.py --mode full --steps 14 9                        # cross-dataset test, binary summary
python run_all.py --steps 5 15 6                                  # board logs, figures and tables
```

The figures exactly as they appear in the paper come from `figures/make_paper_figures.py`, which needs only the files in this
repository: `python figures/make_paper_figures.py` (Figs. 2–7) and `--fig1` for Fig. 1. See `figures/README.md`.

### Checking an export

```bash
python tools/verify_exact_export.py --dataset nf_ton_iot_v2
```

This loads the saved model and runs every check listed above. It also builds three batches of 100,000 inputs: on and next to every
cut, the same values with about 10% replaced by NaN or ±inf, and random float32 bit patterns. Each batch is compared across
scikit-learn, the exact C logic and the legacy C logic, and the compiled header is checked too if a compiler is found.

### Fresh-split replication

```bash
python tools/fresh_split_replication.py --dataset nf_ton_iot_v2 --seeds 7 101 2027
```

Our development runs printed test scores on the seed-42 split, so we wanted to know whether the reported number was optimistic.
This script reruns the whole selection on new splits: the fixed configuration with K chosen on validation, plus a 95-configuration
random search in the restricted space. It picks the winner on validation only and scores each new test split once. The rules
go to `protocol.json` before the first fit. Expect 20–40 minutes per seed and 6–7 GB of free RAM. Our run is in
[`evidence/fresh_split_replication`](evidence/fresh_split_replication/nf_ton_iot_v2).

### On the ESP32

1. Notebook 03 writes `results/esp32_export/<dataset>/esp32_dt_test/` with `esp32_dt_model.h`, `esp32_test_vectors.h` and
   `esp32_dt_test.ino`.
2. Open the sketch in the Arduino IDE (board *ESP32 Dev Module*, Arduino-ESP32 core 3.x), upload it and wait for `#DONE` in the
   Serial Monitor (115200 baud).
3. Save the whole output as `esp32/serial_logs/<dataset>_serial.txt`, or use
   `python tools/capture_serial.py --port COM5 --dataset nf_ton_iot_v2`.
4. Run `python run_all.py --steps 5 6`.

The sketch times only `predict_esp32_dt()` and prints after timing ends (an early version also timed the serial output and looked
about 50 times slower). The board used for the paper was an ESP32-D0WD-V3 at 240 MHz with Arduino-ESP32 core 3.3.11, default
build settings.

### On an Arduino Uno (optional)

The Uno (ATmega328P, 32 KB flash, 2 KB RAM, no FPU) holds trees of up to about 2,000 nodes, for example the binary NF-ToN-IoT-v2
model or the NF-UNSW-NB15-v3 model. Its export keeps every array in flash (`PROGMEM`) and stores only the raw features the tree uses.

```bash
python run_all.py --mode full --datasets nf_ton_iot_v2_binary --steps 3     # if notebook 03 has not run for it
python tools/export_uno.py --dataset nf_ton_iot_v2_binary
# flash uno_dt_test.ino (board: Arduino Uno) and save the output as esp32/serial_logs/<dataset>_uno_serial.txt
python tools/analyze_uno_log.py --dataset nf_ton_iot_v2_binary --sketch-bytes <"Sketch uses" value>
```

## Known limits

- The results describe a per-network flow classifier. On unseen source hosts the multi-class MCC drops to 0.77, and models do not
  transfer between testbeds (see the paper, Sections IV-I and IV-J).
- NF-UNSW-NB15-v3 is separable by a single TTL threshold, so its binary result says little about detection ability.
- The node format stops at 32,767 nodes. Larger trees would need 32-bit child indices, which are not implemented.
- The exact export is implemented and verified for single trees only.

## Citation

```bibtex
@article{aitmessaad2026faithtree,
  title   = {FaithTree: Exact and Leakage-Aware Deployment of Decision-Tree Intrusion Detection on ESP32 Microcontrollers},
  author  = {Ait Messaad, Badr and Motahhir, Saad},
  journal = {IEEE Internet of Things Journal},
  year    = {2026},
  note    = {Submitted}
}
```

## License

MIT, see [LICENSE](LICENSE).
