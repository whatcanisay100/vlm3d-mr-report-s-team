#!/usr/bin/env python3
"""Extract frozen representations from an authorized ordered local manifest."""
import argparse,csv,importlib.util,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=('train','val'),required=True)
    args=p.parse_args()
    import numpy as np,torch
    from .preprocess import preprocess_volume
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('w045_training_runtime',root/'runtime/entrypoint_rg.py')
    entry=importlib.util.module_from_spec(spec);sys.modules[spec.name]=entry;spec.loader.exec_module(entry)
    entry.APP=root/'runtime/app';entry.WEIGHTS=args.weights.resolve()
    _,_,base,_=entry.load_chain()
    if not torch.cuda.is_available():raise RuntimeError('CUDA worker required')
    device=torch.device('cuda:0');encoder,architecture,models=entry.load_vision_models(base,device)
    with args.manifest.open() as f:rows=list(csv.DictReader(f))
    uids=[x['study_uid'] for x in rows]
    if not rows or len(set(uids))!=len(uids):raise ValueError('Empty/duplicate manifest')
    args.output.mkdir(parents=True,exist_ok=True)
    names=[args.output/f'{arm}_{args.split}.npy' for arm in ('bmvp','fomo')]
    uid_path=args.output/f'{args.split}_uids.npy'
    if any(x.exists() for x in [*names,uid_path]):raise FileExistsError('Do not overwrite existing features')
    out=[np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(rows),dim)) for path,dim in zip(names,(36864,1024))]
    for index,row in enumerate(rows):
        volumes=[]
        for modality in ('t1','flair','t2'):
            mask=row.get(modality+'_mask_path') or None
            volumes.append(preprocess_volume(row[modality+'_path'],mask,use_brain_mask=True))
        values=np.stack(volumes)[None]
        out[0][index]=entry.extract_bmvp(base,encoder,values,device)[0]
        out[1][index]=entry.extract_fomo(architecture,models,values,device)[0]
        if index%100==0:print('extracted',index+1,'/',len(rows),flush=True)
    for value in out:value.flush()
    # Committed UID index is written only after all features finish.
    np.save(uid_path,np.asarray(uids,dtype=str),allow_pickle=False)

if __name__=='__main__':main()
