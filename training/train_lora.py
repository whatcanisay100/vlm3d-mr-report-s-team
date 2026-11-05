#!/usr/bin/env python3
"""Public path adapter for the original language-training computation."""
import argparse,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--adapter',type=Path,required=True)
    args=p.parse_args()
    if args.adapter.exists():raise FileExistsError('Do not overwrite an existing adapter')
    from . import lora_core as core
    core.DATA=args.data
    sys.argv=[sys.argv[0],'--model',args.model,'--adapter',str(args.adapter),'--epochs','2']
    core.main()

if __name__=='__main__':main()
