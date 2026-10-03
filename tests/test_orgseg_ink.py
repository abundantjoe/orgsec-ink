import os, sys, json, subprocess, numpy as np, cv2, zarr
ROOT = os.path.join(os.path.dirname(__file__), ".."); PY = sys.executable
def run(script, *args):
    return subprocess.run([PY, os.path.join(ROOT, "orgseg_ink", script), *map(str, args)], capture_output=True, text=True, check=True).stdout

def test_agree_identical_maps_score_one_and_noise_scores_near_zero(tmp_path):
    rng = np.random.default_rng(0); a = (rng.random((1600, 1600)) * 255).astype(np.uint8); a[a == 0] = 1
    b = (rng.random((1600, 1600)) * 255).astype(np.uint8); b[b == 0] = 1   # 1600 px at 8.64 um = 1.4 cm: ~770 independent 0.5 mm cells
    cv2.imwrite(str(tmp_path / "a.png"), a); cv2.imwrite(str(tmp_path / "a2.png"), a); cv2.imwrite(str(tmp_path / "b.png"), b)
    same = json.loads(run("agree.py", tmp_path / "a.png", tmp_path / "a2.png", 8.64, tmp_path / "s.json"))
    diff = json.loads(run("agree.py", tmp_path / "a.png", tmp_path / "b.png", 8.64, tmp_path / "d.json"))
    assert same["r_true"] > 0.99 and abs(diff["r_true"]) < 0.15 and abs(diff["control_max"]) < 0.15

def test_rowscore_prefers_periodic_rows_over_noise(tmp_path):
    pxmm = 53.4; H = W = 1068; y = np.arange(H)[:, None]
    rows = (40 + 150 * (np.sin(2 * np.pi * y / (5.0 * pxmm)) > 0.6)).astype(np.uint8) * np.ones((1, W), np.uint8)  # 5 mm pitch
    noise = (np.random.default_rng(1).random((H, W)) * 200 + 20).astype(np.uint8)
    cv2.imwrite(str(tmp_path / "rows.png"), rows); cv2.imwrite(str(tmp_path / "noise.png"), noise)
    out = [json.loads(l) for l in run("rowscore.py", pxmm, tmp_path / "rows.png", tmp_path / "noise.png").splitlines()]
    assert out[0]["score"] > 5 * out[1]["score"] and 4.0 < out[0]["pitch_mm"] < 6.0

def test_pieces_splits_two_sheets_at_a_dark_seam(tmp_path):
    L, H, W = 31, 400, 400; vol = np.zeros((L, H, W), np.uint8); c = np.full((H, W), 180, np.uint8); c[:, 190:210] = 10  # 20 px dark seam
    vol[L // 2] = c; zarr.save(str(tmp_path / "r.zarr"), vol)
    out = run("pieces.py", tmp_path / "r.zarr", tmp_path / "p", 0.01, 8.64); info = json.load(open(tmp_path / "p_pieces.json"))
    assert len(info["pieces"]) == 2
