#!/usr/bin/env bash
# Re-run the two-model agreement test on the included half-resolution maps (no model, no docker needed).
set -e; cd "$(dirname "$0")/.."; mkdir -p out
for i in 13 15; do python orgseg_ink/agree.py results/maps_half/tx1203_i${i}_ink9um_forward.png results/maps_half/tx1203_i${i}_gpB_forward.png 18.724 out/agree_tx1203_i$i.json; done
python orgseg_ink/rowscore.py 26.7 results/maps_half/tx1203_i13_ink9um_forward.png results/maps_half/sv0800_i1_ink9um_forward.png
