# Part of the orgseg-ink package (MIT).
"""DECL-FL-4 / DECL-FL-4-A1 GP-model inference on one C2-NT render (shared fleet copy of pairs0139c gp_infer.py; numerics unchanged). Numbers only; no image is viewed.
Model: the 2023 Grand Prize TimeSformer (Youssef Nader, Grand Prize team; MIT), as hosted by the organizers on
Hugging Face scrollprize/timesformer_GP_scroll1 @61b81d31 (model.safetensors), built with villa 41701aa
ink-detection/optimized_inference/model_timesformer.py RegressionPLModel(num_frames=26) and loaded STRICTLY.
Inference: villa inference.run_inference (unchanged code) with CFG.tile_size=size=64, CFG.stride=32, no TTA, model.eval(),
sigmoid, bilinear 4x4->64, Hann blending, clip 0..200 then /200. Reduce as villa processing.reduce_partitions:
pred / clip(count, 1e-6), clip 0..1, *255, astype uint8.
Deviation, recorded: villa predict_fn runs torch.autocast(device_type='cpu', enabled=True) (bf16 on CPU). Here autocast is
replaced by a no-op (fp32, the weights' dtype): measured 2x faster on this CPU; fp32-vs-bf16 agreement measured by 'check'.
Layers: render slices 2..27 of 31 (surface = slice 15, the 14th of 26). Variant A: native voxel. Variant B: whole
31-slice stack resampled trilinearly (scipy.ndimage.zoom order 1) by FL_VOX_UM/7.91 in z, y, x (8.64 um: 1.0923); the 26 central slices of the
result; the prediction is resized back to the native piece canvas (cv2 INTER_AREA).
Usage: gp_infer.py run RENDER_ZARR OUT_TIF A|B forward|reverse ;  gp_infer.py check RENDER_ZARR OUT_JSON"""
import sys, os, json, time, contextlib, tempfile, hashlib
import numpy as np, torch, zarr, tifffile, cv2
VILLA = os.environ.get("GP_VILLA", os.path.join(os.environ.get("WORK", "work"), "villa_gp/ink-detection/optimized_inference")); sys.path.insert(0, VILLA)
import inference as VI
from model_timesformer import RegressionPLModel, TimeSformerWrapper
from safetensors.torch import load_file
MODEL = os.environ.get("GP_MODEL", os.path.join(os.environ.get("WORK", "work"), "gp/model.safetensors")); SCALE = float(os.environ["FL_VOX_UM"]) / 7.91
def model():
    m = RegressionPLModel(pred_shape=(1, 1), num_frames=26); m.load_state_dict(load_file(MODEL), strict=True); m.eval(); return m
def stack(rz):
    z = zarr.open(rz, mode="r"); a = z["0"] if hasattr(z, "keys") and "0" in list(z.keys()) else z
    a = np.asarray(a[:]); assert a.ndim == 3 and a.shape[0] == 31 and a.dtype == np.uint8, (a.shape, a.dtype); return a  # (31,H,W)
def layers(a, variant):
    if variant == "A": return np.ascontiguousarray(np.transpose(a[2:28], (1, 2, 0)))
    from scipy import ndimage
    b = ndimage.zoom(a, SCALE, order=1, prefilter=False)
    c0 = (b.shape[0] - 26) // 2
    return np.ascontiguousarray(np.transpose(b[c0:c0 + 26], (1, 2, 0)))
def infer(L, m, reverse, fp32=True):
    VI.CFG.tile_size = 64; VI.CFG.size = 64; VI.CFG.stride = 32; VI.CFG.batch_size = 64; VI.CFG.workers = 0
    VI.CFG.num_parts = 1; VI.CFG.part_id = 0; VI.CFG.zarr_output_dir = tempfile.mkdtemp(dir=os.path.dirname(MODEL))
    orig = torch.autocast
    if fp32: VI.torch.autocast = lambda *a, **k: contextlib.nullcontext()
    try:
        r = VI.run_inference(L, TimeSformerWrapper(m, torch.device("cpu")),
                             torch.device("cpu"), is_reverse_segment=reverse)
    finally:
        VI.torch.autocast = orig
    p = np.asarray(zarr.open(r["mask_pred"], mode="r")[:]); c = np.asarray(zarr.open(r["mask_count"], mode="r")[:])
    import shutil; shutil.rmtree(VI.CFG.zarr_output_dir, ignore_errors=True)
    return (np.clip(p / np.clip(c, 1e-6, None), 0, 1) * 255).astype(np.uint8)
if __name__ == "__main__":
    torch.set_num_threads(4); mode = sys.argv[1]; a = stack(sys.argv[2]); H, W = a.shape[1:]; m = model()
    if mode == "run":
        out, variant, direction = sys.argv[3], sys.argv[4], sys.argv[5]; t = time.time()
        L = layers(a, variant); del a
        u8 = infer(L, m, direction == "reverse")
        if variant == "B": u8 = cv2.resize(u8, (W, H), interpolation=cv2.INTER_AREA)
        assert u8.shape == (H, W); tifffile.imwrite(out, u8)
        print(json.dumps(dict(variant=variant, direction=direction, scale_B=round(SCALE, 5), in_layers_shape=list(L.shape), out_shape=list(u8.shape),
                              seconds=round(time.time() - t, 1), mean=round(float(u8.mean()), 3), frac_ge128=round(float((u8 >= 128).mean()), 5),
                              sha256=hashlib.sha256(open(out, "rb").read()).hexdigest())))
    else:  # check: fp32 vs villa's bf16 autocast on a 640 x 640 central crop, variant A forward
        cy, cx = H // 2 - 320, W // 2 - 320; L = layers(a[:, cy:cy + 640, cx:cx + 640], "A")
        f = infer(L, m, False, True).astype(float); b = infer(L, m, False, False).astype(float); v = L.any(-1)
        d = np.abs(f - b)[v]
        json.dump(dict(crop=[cy, cx, 640, 640], pearson=round(float(np.corrcoef(f[v], b[v])[0, 1]), 6), mean_abs_diff=round(float(d.mean()), 4),
                       p99_abs_diff=float(np.percentile(d, 99)), max_abs_diff=float(d.max()),
                       frac_ge128_fp32=round(float((f[v] >= 128).mean()), 5), frac_ge128_bf16=round(float((b[v] >= 128).mean()), 5)),
                  open(sys.argv[3], "w"), indent=1)
        print(open(sys.argv[3]).read())
