#!/usr/bin/env python3
"""Recreate label/report pairs from authorized local MR-RATE resources."""
import argparse,csv,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--reports',type=Path,required=True,help='Directory of batch*_reports.csv')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    with args.splits.open() as f:
        # The historical preparation iterated a Python set. It did not retain
        # a portable ordering seed; do not promise byte-identical retraining.
        train={r['study_uid'] for r in csv.DictReader(f) if r['split']=='train'}
    reports={}
    for path in sorted(args.reports.glob('batch*_reports.csv')):
        with path.open() as f:
            for row in csv.DictReader(f):
                if row['study_uid'] in train:
                    reports[row['study_uid']]=(str(row.get('clinical_information',''))[:300],str(row.get('report','')))
    with args.labels.open() as f:
        reader=csv.DictReader(f);names=[x for x in reader.fieldnames if x!='study_uid']
        labels={row['study_uid']:[k for k in names if row[k] in ('1','1.0')] for row in reader}
    with args.output.open('x') as f:
        count=0
        for uid in train:
            if uid not in reports or uid not in labels:continue
            clinical,report=reports[uid]
            if not report.strip():continue
            labs=labels[uid]
            user=f"Clinical History: {clinical or 'Not provided.'}\n\nAbnormal findings: {', '.join(labs) if labs else 'None'}\n\nWrite the complete brain MRI report with Technique, Findings, and Impression sections."
            f.write(json.dumps({'uid':uid,'user':user,'assistant':report.strip()})+'\n');count+=1
    print('Wrote',count,'local training pairs; do not redistribute this file.')

if __name__=='__main__':main()
