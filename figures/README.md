# Figures

`make_paper_figures.py` draws the figures of the paper from files in this repository (all seven exactly as printed). No dataset is needed.

| Fig. | Output file | Source of the numbers |
|---|---|---|
| 1 | `fig_framework_icons.png` | none (pipeline diagram), built by `fig1/build_fig1.js` |
| 2 | `fig_exact_threshold_col.png` | none (schematic) |
| 3 | `fig_latency.png` | `esp32/serial_logs/nf_ton_iot_v2_serial.txt`, `nf_unsw_nb15_v3_serial.txt` |
| 4 | `fig_exporter_board_col.png` | `esp32/serial_logs/toolchain_prod_*.txt` |
| 5 | `fig_confusion.png` | `data/confusion_matrix_saved_model.csv` (notebook 02) |
| 6 | `fig_tuning.png` | `data/tab07_optimizer_comparison.csv` (notebook 04) |
| 7 | `fig_unseen_host.png` | `data/generalization_per_class.csv` (notebook 12) |

The three CSV files in `data/` are copies of the result tables written by the notebooks.

## Figs. 2–7

```bash
python figures/make_paper_figures.py
```

The figures are written to `figures/out/` as PNG (300 dpi) and PDF. Only numpy, pandas and matplotlib are needed.

## Fig. 1

Fig. 1 is built as a PowerPoint slide, so it needs Node.js 18 or newer, LibreOffice and PyMuPDF:

```bash
pip install pymupdf
python figures/make_paper_figures.py --fig1
```

The first run installs the Node packages of `figures/fig1/` by itself (`npm install`).

This writes `fig1_pipeline.pptx`, which you can also open and edit in PowerPoint, and the cropped `fig_framework_icons.png`.
If LibreOffice is not on the PATH, pass `--soffice "C:\Program Files\LibreOffice\program\soffice.exe"`.
