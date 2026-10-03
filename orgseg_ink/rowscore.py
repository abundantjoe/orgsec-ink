#!/usr/bin/env python3
# Part of the orgseg-ink package (MIT).
# DECL-FL-5 row-structure score of an ink map: local-mean removed (sigma 2 mm); for theta in -45..45 step 3 deg the row
# projection (valid-weighted) is detrended, Hann-windowed, FFT'd; score(theta) = max power at line pitch 3.5-8 mm / median
# power at 1.5-15 mm; score = max over theta. usage: rowscore.py <px_per_mm> <png>... [--mask m.png --piece k] (json lines)
import sys,json,numpy as np,cv2
args=sys.argv[1:]; pxmm=float(args[0]); files=[a for a in args[1:] if not a.startswith('--')]
mask=None; piece=None
if '--mask' in args: mask=cv2.imread(args[args.index('--mask')+1],0)
if '--piece' in args: piece=int(args[args.index('--piece')+1])
def score(img,valid):
    f=img.astype(np.float32)/255; bg=cv2.GaussianBlur(f*valid,(0,0),2*pxmm)/np.maximum(cv2.GaussianBlur(valid.astype(np.float32),(0,0),2*pxmm),1e-3)
    hp=(f-bg)*valid; H,W=hp.shape; best=(0,0,0)
    lo,hi=int(3.5*pxmm),int(8*pxmm); rlo,rhi=int(1.5*pxmm),int(15*pxmm)
    for th in range(-45,46,3):
        M=cv2.getRotationMatrix2D((W/2,H/2),th,1); r=cv2.warpAffine(hp,M,(W,H)); v=cv2.warpAffine(valid.astype(np.float32),M,(W,H))
        cnt=v.sum(1); ok=cnt>=0.3*cnt.max(); 
        if ok.sum()<lo*2: continue
        prof=np.where(ok,(r*v).sum(1)/np.maximum(cnt,1),0); prof=prof[ok]; prof=prof-prof.mean(); n=len(prof)
        P=np.abs(np.fft.rfft(prof*np.hanning(n)))**2; fr=np.fft.rfftfreq(n); per=np.where(fr>0,1/np.maximum(fr,1e-9),np.inf)
        band=(per>=lo)&(per<=hi); ref=(per>=rlo)&(per<=rhi)
        if band.sum()<1 or ref.sum()<3: continue
        s=P[band].max()/max(np.median(P[ref]),1e-12)
        if s>best[0]: best=(float(s),th,float(per[band][P[band].argmax()]/pxmm))
    return best
for fn in files:
    a=cv2.imread(fn,0); valid=(a>0).astype(np.float32)
    if mask is not None and piece is not None: valid=valid*(mask==piece)
    s,th,pitch=score(a,valid); print(json.dumps({'file':fn,'piece':piece,'score':round(s,2),'theta':th,'pitch_mm':round(pitch,2),'valid_frac':round(float(valid.mean()),3)}))
