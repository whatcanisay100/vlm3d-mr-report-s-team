#!/usr/bin/env python3
"""Materialize the deployed tensor-equivalent FOMO states from upstream."""
import argparse,hashlib
from pathlib import Path

EXPECTED='2bdd1f19b0759cb300b04ce61808feb7d67ae3c28769331355af7954c406aebf'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    digest=hashlib.sha256()
    with args.checkpoint.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):digest.update(block)
    if digest.hexdigest()!=EXPECTED:raise RuntimeError('Not the original FOMO initializer')
    if args.output.exists():raise FileExistsError('Preserve existing output; choose a new directory')
    import torch
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    state={k.removeprefix('model.'):v for k,v in payload['state_dict'].items()
           if k.startswith(('model.encoder.','model.conv_mu_shared.','model.conv_mu_modality.'))}
    if len(state)!=44:raise RuntimeError('Unexpected encoder/latent state schema')
    args.output.mkdir(parents=True)
    for seed,epoch in ((17,7),(42,9),(73,9)):
        torch.save({'seeds':[seed],'selected_epochs':[epoch],'models':[state]},args.output/f'seed{seed}.pt')
    print('Created three tensor-equivalent FOMO states. They are NOT three learned encoders.')

if __name__=='__main__':main()
