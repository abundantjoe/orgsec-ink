#!/usr/bin/env bash
# DECL-FL-6: ink_9um on ORGANIZER segments. JOB sv0800: PHerc0800 organizer surface volumes (31 layers, 8.64 um) copied
# locally; JOB tx1203: PHerc1203 organizer tifxyz rendered here (C2-NT, 31 layers, 9.362 um). Both directions, no TTA.
# usage: orgseg_run.sh <sv0800|tx1203> <IDX>     outputs: logs/fl4t/<JOB>/i<IDX>/ (DONE / FAILED markers)
set -uo pipefail
JOB=$1; I=$2; REPO=${REPO:-$(pwd)}; cd "$REPO"
BR=$(git rev-parse --abbrev-ref HEAD); K=$REPO/tools/fl4; W=${WORK:-work}/fl4t/${JOB}_i$I; mkdir -p $W ${WORK:-work}/own; O=$REPO/logs/fl4t/$JOB/i$I; mkdir -p $O
CKPT=${WORK:-work}/ckpt/ink_9um/hybrid_3d2d-seed42/step-075000.pth; B=vesuvius-challenge-open-data
log(){ echo "[$(date -u +%FT%TZ)] $*" >> $O/pipeline.log; }
proxy_refresh(){ local p; p=$(grep -o 'http://127\.0\.0\.1:[0-9]\+' /nonexistent 2>/dev/null | head -1)
  [ -n "$p" ] && [ "$p" != "${HTTPS_PROXY:-}" ] && export HTTPS_PROXY=$p https_proxy=$p DOCKER_HTTPS_PROXY=$p; return 0; }
push(){ flock ${WORK:-work}/own/git.lock -c "git add logs/fl4t/$JOB/i$I >/dev/null 2>&1; git diff --cached --quiet || git commit -q -m "$JOB i$I: $1""
  for d in 2 4 8 16 32 64; do flock ${WORK:-work}/own/git.lock git push -q origin HEAD:$BR 2>>$W/push.err && return 0
    flock ${WORK:-work}/own/git.lock git pull -q --rebase origin $BR 2>>$W/push.err; sleep $d; done; log "WARN push failed: $1"; }
fail(){ log "FAILED: $1"; echo "$1" > $O/FAILED; push "FAILED $1"; exit 1; }
[ -f $O/DONE ] && exit 0; rm -f $O/FAILED; proxy_refresh
if [ ! -x ${PY:-python} ] || [ ! -x ${INK_PY:-python} ] || [ ! -f $CKPT ] || { [ $JOB = tx1203 ] && { [ ! -x ${WORK:-work}/bin/vc ] || ! docker info >/dev/null 2>&1; }; }; then
  log "env: setup_env.sh"; bash $REPO/docker/env/setup_env.sh > $W/setup_env.log 2>&1 || { tail -40 $W/setup_env.log > $O/env_fail.log.tail; fail "setup_env"; }; fi
R=$W/sv.zarr
if [ ! -f $W/sv.ok ]; then rm -rf $R; log "fetch $JOB i$I"
  if [ $JOB = sv0800 ]; then
    ${PY:-python} - <<PY || fail "sv fetch"
import s3fs,json,os; fs=s3fs.S3FileSystem(anon=True); B='$B'
segs=sorted(x.split('/')[-1] for x in fs.ls(f'{B}/PHerc0800/segments')); s=segs[$I]
sv=f'{B}/PHerc0800/segments/{s}/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr'
os.makedirs('$R',exist_ok=True); fs.get(sv+'/.zattrs','$R/.zattrs'); fs.get(sv+'/.zgroup','$R/.zgroup'); fs.get(sv+'/0','$R/0',recursive=True)
json.dump({'scroll':'PHerc0800','segment':s,'surface_volume':'s3://'+sv,'mesh':'s3://'+f'{B}/PHerc0800/segments/{s}/mesh','um':8.64},open('$O/source.json','w'),indent=1); print('fetched',s)
PY
  else
    ${PY:-python} - <<PY || fail "tifxyz fetch"
import s3fs,json,os; fs=s3fs.S3FileSystem(anon=True); B='$B'
segs=sorted(x.split('/')[-1] for x in fs.ls(f'{B}/PHerc1203/segments/raw')); s=segs[$I]; d='$W/seg'; os.makedirs(d,exist_ok=True)
for f in ['x.tif','y.tif','z.tif','meta.json']: fs.get(f'{B}/PHerc1203/segments/raw/{s}/{f}',f'{d}/{f}')
m=json.load(open(f'{d}/meta.json')); json.dump({'scroll':'PHerc1203','segment':s,'tifxyz':'s3://'+f'{B}/PHerc1203/segments/raw/{s}','meta':m,'um':9.362},open('$O/source.json','w'),indent=1); print('fetched',s,{k:m[k] for k in list(m)[:6]})
PY
    CT=s3://$B/PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr/; log "render"
    ( cd $W && ${WORK:-work}/bin/vc vc_render_tifxyz -v $CT --remote-url $CT --segmentation $W/seg --scale 1 --group-idx 0 --num-slices 31 --slice-step 1 \
      --voxel-size 9.362 --voxel-unit micrometer --flip-normals --cache-gb 3 --zarr-output $R > $W/render.log 2>&1 ) || { tail -5 $W/render.log > $O/render.log.tail; fail "render"; }
    mkdir -p $O/tifxyz; cp $W/seg/*.tif $W/seg/meta.json $O/tifxyz/
  fi
  touch $W/sv.ok; push "source"; fi
[ -f $O/${JOB}_i${I}_pieces.json ] || { ${PY:-python} $K/pieces.py $R $O/${JOB}_i$I 0.25 $( [ $JOB = sv0800 ] && echo 8.64 || echo 9.362 ) >> $O/pipeline.log 2>&1 || log "pieces errored"; }
if [ ! -f $O/ink9um_forward.png ] || [ ! -f $O/ink9um_reverse.png ]; then log "ink_9um both"; t0=$SECONDS
  ( cd $W && OMP_NUM_THREADS=4 ${INK_PY:-python} -m vesuvius.ink_detection.inference.infer $R $CKPT $W/ink.tif --direction both --batch-size 4 --num-workers 1 --no-compile > $W/ink.log 2>&1 ) || { tail -5 $W/ink.log > $O/ink_fail.log.tail; fail "ink_9um"; }
  ${PY:-python} - <<PY || fail "ink png"
import tifffile,numpy as np,cv2,json
for src,dst in [('$W/ink.tif','$O/ink9um_forward.png'),('$W/ink_reverse.tif','$O/ink9um_reverse.png')]:
    t=tifffile.imread(src).astype(np.float32); p=t/(255.0 if t.max()>1.5 else 1.0)
    cv2.imwrite(dst,np.clip(p*255,0,255).astype(np.uint8),[cv2.IMWRITE_PNG_COMPRESSION,6])
    json.dump({'src':src,'shape':list(t.shape),'min':float(t.min()),'max':float(t.max()),'frac_ge05':float((p>=0.5).mean()),'frac_ge08':float((p>=0.8).mean())},open(dst[:-4]+'.json','w'))
PY
  log "ink_9um done $((SECONDS-t0)) s"; fi
touch $O/DONE; push "DONE"; rm -rf $R
