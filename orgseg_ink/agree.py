#!/usr/bin/env python3
# Part of the orgseg-ink package (MIT).
# DECL-FL-6 cross-check: do two independent ink models mark the same places? Pearson r between 0.5 mm-smoothed maps
# (valid pixels only) vs controls: the second map rotated 90/180/270 deg and flipped (same statistics, wrong geometry).
# usage: agree.py <mapA.png> <mapB.tif|png> <um_per_px> <out.json>
import sys,json,numpy as np,cv2,tifffile
a=cv2.imread(sys.argv[1],0).astype(np.float32); fb=sys.argv[2]; um=float(sys.argv[3])
b=(tifffile.imread(fb) if fb.endswith('.tif') else cv2.imread(fb,0)).astype(np.float32)
if b.shape!=a.shape: b=cv2.resize(b,(a.shape[1],a.shape[0]),interpolation=cv2.INTER_AREA)
v=(a>0)&(b>0); s=0.5*1000/um
def sm(x): return cv2.GaussianBlur(x,(0,0),s)
A=sm(a*v)/np.maximum(sm(v.astype(np.float32)),1e-3); B=sm(b*v)/np.maximum(sm(v.astype(np.float32)),1e-3)
def r(x,y,m): x=x[m]-x[m].mean(); y=y[m]-y[m].mean(); return float((x*y).sum()/np.sqrt((x*x).sum()*(y*y).sum()+1e-9))
res={'r_true':r(A,B,v),'controls':{}}
for name,T in [('rot90',lambda z:np.rot90(z,1)),('rot180',lambda z:np.rot90(z,2)),('rot270',lambda z:np.rot90(z,3)),('flipud',np.flipud),('fliplr',np.fliplr)]:
    Bt=T(B); vt=T(v)
    if Bt.shape==A.shape: res['controls'][name]=r(A,Bt,vt&v)
res['control_max']=max(res['controls'].values()); res['valid_px']=int(v.sum()); json.dump(res,open(sys.argv[4],'w'),indent=1); print(json.dumps(res))
