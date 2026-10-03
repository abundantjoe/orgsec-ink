#!/usr/bin/env bash
# Part of the orgseg-ink package (MIT).
# DECL-FL-4 / -A1: CPU environment for the Grand Prize TimeSformer (organizer copy HF scrollprize/timesformer_GP_scroll1
# @61b81d31, MIT, by Youssef Nader for the 2023 Grand Prize team) run through villa 41701aa optimized_inference.
# Idempotent. Creates ${GP_VENV:-$HOME/.venvs/gp} (venv), ${WORK:-work}/villa_gp (sparse villa at 41701aa), ${WORK:-work}/gp/model.safetensors (sha-checked).
set -euo pipefail
GPSHA=490a98f9491e1180274ed3a0c0a9c611d73a0109c0e0c0fbba1097562a972488
HFREV=61b81d31ddb93c89a2b3b00c66769d18bd96ccb2; VREV=41701aa
mkdir -p ${WORK:-work}/gp
if [ ! -x ${GP_VENV:-$HOME/.venvs/gp}/bin/python ] || ! ${GP_VENV:-$HOME/.venvs/gp}/bin/python -c "import torch, timesformer_pytorch, albumentations, pytorch_lightning, zarr, safetensors, cv2, tifffile, scipy, fsspec" 2>/dev/null; then
  uv venv -q --python 3.11 ${GP_VENV:-$HOME/.venvs/gp}
  VIRTUAL_ENV=${GP_VENV:-$HOME/.venvs/gp} uv pip install -q --index-url https://download.pytorch.org/whl/cpu torch
  VIRTUAL_ENV=${GP_VENV:-$HOME/.venvs/gp} uv pip install -q timesformer-pytorch==0.4.1 einops albumentations pytorch-lightning "zarr>=3" numcodecs fsspec \
    safetensors tifffile opencv-python-headless scipy tqdm psutil numpy
fi
if [ ! -f ${WORK:-work}/villa_gp/ink-detection/optimized_inference/inference.py ]; then
  rm -rf ${WORK:-work}/villa_gp
  git clone -q --filter=blob:none --no-checkout https://github.com/ScrollPrize/villa.git ${WORK:-work}/villa_gp
  git -C ${WORK:-work}/villa_gp sparse-checkout set ink-detection/optimized_inference
  git -C ${WORK:-work}/villa_gp checkout -q $VREV
fi
[ "$(git -C ${WORK:-work}/villa_gp rev-parse --short=7 HEAD)" = "$VREV" ] || { echo "villa_gp not at $VREV"; exit 1; }
if ! echo "$GPSHA  ${WORK:-work}/gp/model.safetensors" | sha256sum -c --quiet 2>/dev/null; then
  curl -fsSL -o ${WORK:-work}/gp/model.safetensors.part "https://huggingface.co/scrollprize/timesformer_GP_scroll1/resolve/$HFREV/model.safetensors"
  mv ${WORK:-work}/gp/model.safetensors.part ${WORK:-work}/gp/model.safetensors
  echo "$GPSHA  ${WORK:-work}/gp/model.safetensors" | sha256sum -c --quiet || { echo "GP model sha256 mismatch"; exit 1; }
fi
FL_VOX_UM=8.64 ${GP_VENV:-$HOME/.venvs/gp}/bin/python -c "
import sys; sys.argv=['x']; sys.path.insert(0,'$(dirname "$0")'); import gp_infer as G; m=G.model(); print('GP model strict load OK', sum(p.numel() for p in m.parameters()))"
echo GP_ENV_OK
