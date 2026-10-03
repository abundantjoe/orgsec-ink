#!/usr/bin/env python3
# Part of the orgseg-ink package (MIT).
# DECL-FL-5 sheet-piece map: the render's centre layer, smoothed (sigma 0.5 mm), thresholded by Otsu -> connected on-papyrus
# pieces (>= min_cm2). A layer jump at a crack passes through air; a 0.3 mm opening splits pieces at such thin dark seams. usage: pieces.py <render.zarr> <out_prefix> [min_cm2=0.25] [um_px=8.64]
import sys,json,numpy as np,zarr,cv2
R=zarr.open(sys.argv[1],mode='r'); out=sys.argv[2]; mn=float(sys.argv[3]) if len(sys.argv)>3 else 0.25; um=float(sys.argv[4]) if len(sys.argv)>4 else 8.64
a=R['0'] if isinstance(R,zarr.Group) and '0' in R else R; L=a.shape[0]; c=np.asarray(a[L//2]).astype(np.float32); valid=c>0
pxmm=1000/um; s=cv2.GaussianBlur(c,(0,0),0.1*pxmm); s[~valid]=0
v=s[valid]; thr,_=cv2.threshold(np.clip(v/v.max()*255,0,255).astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU); thr=thr/255*v.max()
on=(s>thr)&valid; k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(0.3*pxmm)|1,)*2); on=cv2.morphologyEx(on.astype(np.uint8),cv2.MORPH_OPEN,k).astype(bool)
n,lab,st,_=cv2.connectedComponentsWithStats(on.astype(np.uint8),8)
keep=[i for i in range(1,n) if st[i,cv2.CC_STAT_AREA]>=mn*100*pxmm*pxmm]; keep.sort(key=lambda i:-st[i,cv2.CC_STAT_AREA])
m=np.zeros(c.shape,np.uint8)
for k,i in enumerate(keep[:254]): m[lab==i]=k+1
cv2.imwrite(out+'_pieces.png',m); cv2.imwrite(out+'_centre.png',np.clip(c/np.percentile(c[valid],99.5)*255,0,255).astype(np.uint8),[cv2.IMWRITE_PNG_COMPRESSION,6])
info={'layers':L,'otsu':float(thr),'valid_frac':float(valid.mean()),'on_frac':float(on.sum()/max(valid.sum(),1)),'pieces':[{'id':k+1,'area_cm2':float(st[i,cv2.CC_STAT_AREA]/(100*pxmm*pxmm)),'bbox':[int(x) for x in st[i,:4]]} for k,i in enumerate(keep[:254])]}
json.dump(info,open(out+'_pieces.json','w'),indent=1); print(json.dumps({k:v for k,v in info.items() if k!='pieces'}),'n_pieces',len(info['pieces']),'largest',[round(p['area_cm2'],2) for p in info['pieces'][:5]])
