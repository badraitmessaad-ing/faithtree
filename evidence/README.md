# Evidence

Small result files that back specific claims in the paper. They were copied from `results/` after the runs, so you can check
the numbers without the datasets. Everything here can be regenerated with the commands below.

## exact_export_verification/

Output of `tools/verify_exact_export.py` for the two models that run on the ESP32 board in the paper.

| File | Model | What it shows |
|---|---|---|
| `nf_ton_iot_v2.json` | 7,073-node tree | 3,536/3,536 nodes pass the cut check and the NaN, ±inf, ±0 and ±FLT_MAX checks; 30/30 features monotone; 0 mismatches for the exact export on 3 × 100,000 adversarial inputs (NumPy logic and compiled C), against 1,270 and 1,065 for the legacy export |
| `nf_unsw_nb15_v3.json` | 1,217-node tree | same checks, 608/608 nodes, 34/34 features; legacy export 901 and 783 mismatches |

```bash
python tools/verify_exact_export.py --dataset nf_ton_iot_v2
python tools/verify_exact_export.py --dataset nf_unsw_nb15_v3
```

The compiler line was shortened to "MinGW gcc (Anaconda), Windows". The original output contained a local path.

## fresh_split_replication/nf_ton_iot_v2/

Our development runs printed test scores on the seed-42 split, so we checked whether the reported test MCC could be optimistic.
`tools/fresh_split_replication.py` repeats the whole model selection on three splits that were never used before (seeds 7, 101 and
2027) and scores each new test split once.

| File | Content |
|---|---|
| `protocol.json` | the rules, written before the first fit (10:58, 14 Oct 2026) |
| `search_log.csv` | all 303 candidates of stage 1 (6 fixed-config K values + 95 random configurations per seed) |
| `finalists.csv` | the 4 finalists per seed, refitted on the full training split and scored on validation |
| `runs.csv` | the winner of each seed and its single test score |
| `summary.json` | aggregate result |
| `console_output.txt` | the terminal output of the run |

Result: test MCC **0.9721 ± 0.0011** over the three new splits, against 0.9722 reported for seed 42. Validation and test MCC differ by
0.0001 on average. The paper's fixed configuration won on seed 101 and was within 0.0015 validation MCC of the winner on the other two.
On seed 2027 the winning tree grew to 34,471 nodes in the final refit on train + validation, above the 32,767-node limit of the export format.
The size limit is checked on the training-split fit, so this tree was selected but could not be exported.

```bash
python tools/fresh_split_replication.py --dataset nf_ton_iot_v2 --seeds 7 101 2027
```

About 20–40 minutes per seed on a desktop PC; it needs 6–7 GB of free RAM.
