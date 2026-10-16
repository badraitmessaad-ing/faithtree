"""
capture_serial.py -- save the ESP32 Serial output to  esp32/serial_logs/<dataset>_serial.txt
(alternative to copy-pasting from the Arduino Serial Monitor; close the Serial Monitor first).

    pip install pyserial
    python tools/capture_serial.py --port COM5 --dataset nf_unsw_nb15_v3
    python tools/capture_serial.py --port /dev/ttyUSB0 --dataset nf_ton_iot_v2 --tag footprint_with_model

Press the board's EN/RESET button right after starting this script so the whole run is captured.
It stops automatically after the line  #DONE  (or after --timeout seconds).
"""
import argparse, time
from pathlib import Path
import serial   # pyserial

ap = argparse.ArgumentParser()
ap.add_argument("--port", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--tag", default="serial", help="serial | footprint_with_model | footprint_without_model")
ap.add_argument("--baud", type=int, default=115200)
ap.add_argument("--timeout", type=int, default=600)
a = ap.parse_args()

out = Path(__file__).resolve().parents[1] / "esp32" / "serial_logs" / f"{a.dataset}_{a.tag}.txt"
out.parent.mkdir(parents=True, exist_ok=True)
t0 = time.time()
with serial.Serial(a.port, a.baud, timeout=1) as s, open(out, "w", encoding="utf-8") as f:
    print("listening on", a.port, "-> press RESET on the board ...")
    while time.time() - t0 < a.timeout:
        line = s.readline().decode("utf-8", errors="ignore")
        if not line:
            continue
        f.write(line); f.flush()
        if "#DONE" in line:
            break
print("saved", out)
