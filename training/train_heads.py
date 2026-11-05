#!/usr/bin/env python3
"""Train/export proxy32 heads with the original frozen numerical routines."""
import argparse,json,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--train-manifest',type=Path,required=True)
    p.add_argument('--val-manifest',type=Path,required=True)
    p.add_argument('--label-map',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import numpy as np,pandas as pd,torch
    from sklearn.decomposition import PCA
    from . import head_core as core
    from .label_mapping import map_labels
    if args.output.exists():raise FileExistsError('Preserve existing bundle')
    train=pd.read_csv(args.train_manifest,dtype={'study_uid':str,'patient_key':str})
    val=pd.read_csv(args.val_manifest,dtype={'study_uid':str,'patient_key':str})
    if len(train)!=82608 or len(val)!=3483:raise ValueError('Expected 82608 train and 3483 natural-val studies')
    if train.study_uid.duplicated().any() or val.study_uid.duplicated().any():raise ValueError('Duplicate study')
    if set(train.study_uid)&set(val.study_uid) or set(train.patient_key)&set(val.patient_key):raise ValueError('Patient/study overlap')
    labels=pd.read_csv(args.labels,dtype={'study_uid':str}).set_index('study_uid',verify_integrity=True)
    names=list(labels.columns)
    if len(names)!=37:raise ValueError('Expected 37 supplied pathology-label columns')
    mapping=json.loads(args.label_map.read_text());keep=mapping['keep32']
    yt,mask=map_labels(labels.loc[train.study_uid].to_numpy(dtype=np.float32),keep,names)
    yv,vmask=map_labels(labels.loc[val.study_uid].to_numpy(dtype=np.float32),keep,names)
    if not np.array_equal(mask,vmask) or mask.sum()!=8:raise ValueError('Masked category drift')
    if not torch.cuda.is_available():raise RuntimeError('Run on a dedicated CUDA worker, not a shared control node')
    device=torch.device('cuda');started=time.time();args.output.mkdir(parents=True)
    for arm in ('bmvp','fomo'):
        xt=np.load(args.features/f'{arm}_train.npy',allow_pickle=False)
        xv=np.load(args.features/f'{arm}_val.npy',allow_pickle=False)
        # Feature order is checked against the local manifest, not assumed.
        for split,frame in (('train',train),('val',val)):
            seen=np.load(args.features/f'{split}_uids.npy',allow_pickle=False).astype(str)
            if not np.array_equal(seen,frame.study_uid.to_numpy()):raise ValueError('Feature/label order mismatch')
        dim=36864 if arm=='bmvp' else 1024
        if xt.shape!=(82608,dim) or xv.shape!=(3483,dim):raise ValueError('Feature dimensions drift')
        if not np.isfinite(xt).all() or not np.isfinite(xv).all():raise ValueError('Nonfinite features')
        mean=xt.mean(0);std=xt.std(0)+1e-6;xt=(xt-mean)/std;xv=(xv-mean)/std
        pre=dict(raw_mean=mean.astype(np.float32),raw_std=std.astype(np.float32))
        if arm=='bmvp':
            reducer=PCA(n_components=2048,random_state=0)
            xt=reducer.fit_transform(xt);xv=reducer.transform(xv)
            pre.update(pca_components=reducer.components_.astype(np.float32),pca_mean=reducer.mean_.astype(np.float32))
            states,_,_,records=core.train_bmvp(xt,yt,xv,yv,~mask,device,started)
            seeds=core.BMVP_SEEDS
        else:
            states,_,_,records=core.train_fomo(xt,yt,xv,~mask,device,started)
            seeds=core.FOMO_SEEDS
        np.savez(args.output/f'{arm}_preprocessor.npz',**pre)
        torch.save(dict(state_dicts=states,seeds=seeds,records=records),args.output/f'{arm}_heads.pt')
    (args.output/'label_map.json').write_text(json.dumps(mapping,indent=2)+'\n')

if __name__=='__main__':main()
