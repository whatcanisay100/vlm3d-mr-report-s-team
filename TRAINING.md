# Training and data reconstruction

Use an authorized MR-RATE copy and a dedicated CUDA worker. No training is
started by installing/importing this release. For training, add pandas,
scikit-learn, datasets and the needed PyTorch/CUDA build to the recorded
runtime environment. Historical head export recorded Python3.10.20,
NumPy2.2.5, pandas2.3.3, scikit-learn1.7.2 and torch2.12.1+cu130; this is not
claimed to be a newly tested universal environment specification.

## 1. Local visual manifests

Create ordered CSVs from the official split/labels/series metadata, with:

```text
study_uid,patient_key,t1_path,flair_path,t2_path,t1_mask_path,flair_mask_path,t2_mask_path
```

Paths point to your authorized local NIfTI files. Use a stable composite
patient key consistently across splits, and preserve the original row order
for comparison. Mask paths can be empty; absent masks use nonzero foreground.
The feature adapter does not select cases using reports or predictions.

The archived visual cohort retained 47,906 audited rows from the first48K
plus 34,702 additional complete labeled rows. It was not an arbitrary
random82K subsample. Final ordered train UID SHA256 (newline joined, no final
newline): `4c783bc959755f5700b803b7d123df26b10743e8ed296f894b01e128b9691fb6`;
natural-val3483 SHA256:
`60f92d2da6409b624ebe811bf0d0a0c205e126464523a1ddd4485cd6e7902ce9`.
Private historical row lists, intermediate CSVs and patient IDs are not
redistributed. The public adapter accepts a locally reconstructed manifest;
it does not pretend to reconstruct the original audited ordering automatically.
If your manifest differs, report it as a retraining variant.

After acquiring upstream weights and preparing the frozen FOMO states as in
README, run each extraction serially or on dedicated compute workers:

```bash
python -m training.extract_features --manifest /data/train.csv --split train \
  --weights /data/weights --output /data/features
python -m training.extract_features --manifest /data/val.csv --split val \
  --weights /data/weights --output /data/features
python -m training.train_heads --features /data/features \
  --train-manifest /data/train.csv --val-manifest /data/val.csv \
  --labels /data/MR-RATE/pathology_labels/mrrate_labels.csv \
  --label-map /data/weights/bundle/label_map.json --output /data/new_bundle
```

Features remain local: `bmvp_{train,val}.npy`, `fomo_{train,val}.npy`,
`{train,val}_uids.npy`. UID indexes are committed only after extraction
finishes; incomplete outputs are not silently treated as valid. There is no
automatic experiment restart, GPU takeover, downloader or hidden background
worker. The head CLI validates dimensions, row counts, uniqueness, patient
disjointness, UID/label alignment and the eight masked classes. Core training
loops are extracted unchanged from the final head-export source.

Historical FOMO auxiliary-head training did not update its frozen encoder.
No new end-to-end encoder fine-tuning is introduced as a substitute here.

## 2. Language adapter

Only the official training split is used to create supplied-label/report pairs:

```bash
python -m training.prepare_language_data \
  --splits /data/MR-RATE/splits.csv \
  --labels /data/MR-RATE/pathology_labels/mrrate_labels.csv \
  --reports /data/MR-RATE/reports --output /data/finetune_train.jsonl
torchrun --nproc_per_node=2 -m training.train_lora \
  --data /data/finetune_train.jsonl --model /data/Qwen2.5-14B-Instruct \
  --adapter /data/new_qwen14b_lora
```

The original language-data preparation iterated a Python set and used pandas
string coercions; exact serialized order/missing-value rendering is not
guaranteed by the clean CSV adapter. The recorded run had88,582 pairs. Keep
the generated JSONL private, record its hash/order in your own experiment,
and disclose deviations. Training targets are supplied reports; deployment
uses predicted labels. This is not joint image-language alignment.

## 3. Acceptance boundary

The released numerical routines preserve historical behavior, including
legacy feature layout, masking, epoch-level schedules and model selection.
The public path/data adapters and Docker recipe have syntax/contract checks,
but a complete fresh GPU retraining and isolated GPU inference run have not
been performed for this cleaned package. Do not label a new run an exact
reproduction solely because it completes. Keep train/validation identity
checks, compare the deployed final weights and document your environment.
