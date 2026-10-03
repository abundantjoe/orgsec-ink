#!/usr/bin/env bash
# DECL-FL-6-A1: GP TimeSformer variant B (forward) on a PHerc1203 organizer segment already in logs/fl4t/tx1203/i<IDX>/tifxyz.
# Re-renders (C2-NT, 31 layers, 9.362 um), runs gp_infer.py, pushes logs/fl4t/tx1203/i<IDX>/gpB_forward.png. Markers GP_DONE/GP_FAILED.
set -uo pipefail; I=$1; REPO=${REPO:-$(pwd)}; cd "$REPO"; BR=$(git rev-parse --abbrev-ref HEAD)
K=$REPO/tools/fl4; W=${WORK:-work}/fl4t/gp_i$I; mkdir -p $W ${WORK:-work}/own; O=$REPO/logs/fl4t/tx1203/i$I; B=vesuvius-challenge-open-data
log(){ echo "[$(date -u +%FT%TZ)] GP $*" >> $O/pipeline.log; }
proxy_refresh(){ local p; p=$(grep -o 'http://127\.0\.0\.1:[0-9]\+' /nonexistent 2>/dev/null | head -1)
  [ -n "$p" ] && [ "$p" != "${HTTPS_PROXY:-}" ] && export HTTPS_PROXY=$p https_proxy=$p DOCKER_HTTPS_PROXY=$p; return 0; }
push(){ flock ${WORK:-work}/own/git.lock -c "git add $O >/dev/null 2>&1; git diff --cached --quiet || git commit -q -m "tx1203 i$I: $1""
  for d in 2 4 8 16 32; do flock ${WORK:-work}/own/git.lock git push -q origin HEAD:$BR 2>>$W/push.err && return 0; flock ${WORK:-work}/own/git.lock git pull -q --rebase origin $BR 2>>$W/push.err; sleep $d; done; }
fail(){ log "FAILED: $1"; touch $O/GP_FAILED; push "GP failed $1"; exit 1; }
[ -f $O/GP_DONE ] && exit 0; rm -f $O/GP_FAILED; proxy_refresh
GPY=${GP_VENV:-$HOME/.venvs/gp}/bin/python; [ -x $GPY ] || GPY=${INK_PY:-python}; [ -f ${WORK:-work}/gp/model.safetensors ] || { bash $K/setup_gp.sh > $W/setup_gp.log 2>&1 || fail "setup_gp"; }
docker info >/dev/null 2>&1 || { bash $REPO/docker/env/setup_env.sh > $W/setup_env.log 2>&1 || fail "setup_env"; }
R=$W/sv.zarr; CT=s3://$B/PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr/
if [ ! -d $R/0 ]; then log "render"; rm -rf $R; ( cd $W && ${WORK:-work}/bin/vc vc_render_tifxyz -v $CT --remote-url $CT --segmentation $O/tifxyz --scale 1 --group-idx 0 --num-slices 31 --slice-step 1 --voxel-size 9.362 --voxel-unit micrometer --flip-normals --cache-gb 3 --zarr-output $R > $W/render.log 2>&1 ) || fail "render"; fi
log "gp B forward"; FL_VOX_UM=9.362 OMP_NUM_THREADS=4 $GPY $K/gp_infer.py run $R $W/gp_fwd.tif B forward > $W/gp.log 2>&1 || fail "gp"
${PY:-python} -c "
import tifffile,numpy as np,cv2,json
t=tifffile.imread('$W/gp_fwd.tif').astype(np.float32); p=t/(255.0 if t.max()>1.5 else 1.0)
cv2.imwrite('$O/gpB_forward.png',np.clip(p*255,0,255).astype(np.uint8),[cv2.IMWRITE_PNG_COMPRESSION,6]); json.dump({'shape':list(t.shape),'frac_ge05':float((p>=0.5).mean())},open('$O/gpB_forward.json','w'))" || fail "png"
touch $O/GP_DONE; log "done"; push "GP B forward"; rm -rf $R
