# S-team: BrainMVP–FOMO Fusion for MRI Report Generation

Final-submission implementation and release documentation for **VLM3D @ MICCAI
2026**, task **MR Report Generation**, submission **`1e7ab392`**, shorthand
**w045**. This describes the submitted label-conditioned system, not subsequent
research prototypes.

* Model assets: https://huggingface.co/whatcanisay100/vlm3d-mr-report-s-team-w045
* Own software: MIT (`LICENSE`). Trained assets: see their model card and
  separate license. Third-party components retain their original terms.
* Research use; not a medical device or a validated clinical decision system.

## Team

| Member | Affiliation |
| --- | --- |
| Xiao Yang | University of Electronic Science and Technology of China, Chengdu, China |
| Jiaqi Gu | Fudan University, Shanghai, China |
| Lei Liu | Fudan University, Shanghai, China |
| Shushang Jiang | Fudan University, Shanghai, China |

No author-contribution or corresponding-author designation is inferred.

## Method

```text
T1w + FLAIR + T2w MRI
   ├─ frozen BrainMVP legacy C8 → train-fitted scaling/PCA → 10 MLP heads
   └─ frozen FOMO representation → train-fitted scaling → 10 MLP heads
                    ↓ mean sigmoid probabilities within each head ensemble
         0.55 × BrainMVP + 0.45 × FOMO; threshold 0.5
                    ↓ predicted abnormality labels
         Qwen2.5-14B-Instruct + trained language LoRA
                    ↓ negation-aware positive-label coverage check
               Technique / Findings / Impression report
```

The language model receives **labels, not MRI tokens**. This is not an
end-to-end visual-token VLM, and retrieval-based report rewriting was not used.

### Important correction to the earlier documentation

The three historical FOMO files were named and previously described as
fine-tuned encoder states. A release audit on 2026-11-05 compared all **44 named
encoder/latent-projection tensors per file** against the original initializer:
**44/44 were identical in each of seeds 17, 42 and 73**. The training loader
froze the encoder, and the corresponding training entrypoint did not re-enable
its gradients. They therefore must not be described as three independently
adapted encoders. The deployed runtime averages three tensor-identical copies;
this behavior is retained for historical reproduction.

This correction does not mean the downstream MLP heads or language LoRA were
untrained. It corrects the encoder-adaptation claim, not a newly measured
performance result. The original files and submitted weights have not been
retrained or replaced. `tools/prepare_frozen_fomo.py` materializes the same
encoder tensors from the original public initializer, without redistributing
that initializer or relabeling it as an S-team-trained model.

## Data and preprocessing

Obtain MR-RATE through its authorized provider:
https://huggingface.co/datasets/Forithmus/MR-RATE . Its access agreement,
confidentiality and redistribution restrictions apply separately. This
repository contains **no MRI, reports, annotations, patient identifiers,
feature caches or per-case predictions**.

Deployed image-only inference selects one T1w, FLAIR and T2w series from each
ZIP deterministically, by case-folded filename then full member path. Atlas,
segmentation, reports, labels and clinical-history text are not inference
inputs. SWI is not used.

Each NIfTI array is converted to float32, nonfinite values become zero, the
nonzero foreground is cropped with a 12-voxel margin, intensities are clipped
at foreground percentiles 0.5/99.5, foreground-z-normalized, clipped to [-5,5],
divided by 5 and resized by first-order interpolation to **128×128×96**.
Tensor order is **T1, FLAIR, T2**. Deployment does not apply a supplied brain
mask or guarantee physical inter-sequence registration.

BrainMVP uses the **submitted legacy** patch embedding and C8 layout:
36,864 dimensions, train-fitted standardization, PCA to 2,048. Do not silently
substitute the later checkpoint-faithful LayerNorm/spatial-layout correction;
the supplied heads are not trained for it. FOMO aggregates deterministic 64³
windows and shared/modality latent means to a 1,024-dimensional representation.

Training feature preprocessing can use available brain masks, as in the
historical training pipeline; it is not represented as identical to deployed
image-only preprocessing. The public feature adapter accepts these mask paths
explicitly. A different series manifest/order or mask availability changes
retraining and is not guaranteed to reproduce the archived tensors.

## Installation and exact inference assets

Use a CUDA worker; the submitted deployment targets one **A100 80 GB**. Do not
run model loading on a shared control node. Frozen runtime requirements are in
`runtime/requirements.txt`. The Docker base digest is pinned in `Dockerfile`.

1. Download the seven release files from the Hugging Face repository into
   `weights/`, preserving `adapter/` and `bundle/`. Verify their bytes/SHA256
   against `WEIGHT_MANIFEST.json`.
2. Obtain **BrainMVP Uniformer** from its authors:
   https://github.com/shaohao011/BrainMVP . Put it at
   `weights/bmvp/BrainMVP_uniformer.pt`.
   SHA256: `93e9fbf71d38ae6b41447410e2d04977959385b980494be8ea15edaa0a31df23`.
3. Obtain the original **FOMO2JOMO** initializer from
   https://github.com/jbanusco/fomo25/releases/tag/v1.0.0 . Do not substitute a
   later release solely because its filename is the same. Expected SHA256:
   `2bdd1f19b0759cb300b04ce61808feb7d67ae3c28769331355af7954c406aebf`.
   Generate the three tensor-equivalent deployment states:

   ```bash
   python tools/prepare_frozen_fomo.py \
     --checkpoint /path/to/fomo25_mmunetvae_pretrained.ckpt \
     --output weights/fomo
   ```

4. Obtain **Qwen/Qwen2.5-14B-Instruct** from its official repository, including
   configuration, tokenizer, shard index and all eight safetensors shards.
   Place these under `weights/qwen14b/`. We do not redistribute the base model.
   The LoRA tensor is unchanged; only its JSON base-model path has been
   normalized to the official model ID to remove a private filesystem path.
5. Obtain the official `evaluation_v2.py` through the challenge's evaluation
   resources / https://github.com/forithmus/VLM3D-Dockers and place it at
   `runtime/app/evaluation_v2.py`. It is an external dependency, not
   redistributed under our MIT license. The build requires SHA256
   `b4af91a1c0eab7cc5935da5956905576daabe67b226bbac6eeb5537e9a9beb37`.
   Its label extractor is used for coverage checking, not ground-truth access.

Build and run after supplying those dependencies:

```bash
docker build -t s-team-w045:1e7ab392 .
# Make output/checkpoint directories writable by the container UID 10053.
docker run --rm --gpus '"device=0"' \
  -v /absolute/input:/input:ro \
  -v /absolute/weights:/weights:ro \
  -v /absolute/output:/output \
  -v /absolute/checkpoint:/checkpoint \
  s-team-w045:1e7ab392
```

The container does not download weights at inference time. Restrict visible
GPUs to one. A model failure is fatal rather than silently made into a normal
report. Incomplete/unusable image inputs have a separate technical-limitation
route. Only complete batches are atomically committed for resumption.

### Input/output

Input: `/input/EXAMPLE_STUDY.zip`, containing
`EXAMPLE_STUDY/img/EXAMPLE_STUDY_{t1w,flair,t2w}-raw-axi.nii.gz`.
This is a synthetic naming example, not a real case.

Output: `/output/predictions.json`:

```json
{"generated_reports":[{"input_image_name":"EXAMPLE_STUDY.nii.gz","report":"Technique: ... Findings: ... Impression: ..."}]}
```

Decoding is FP16, greedy, maximum 1,024 new tokens, prompt truncation 2,048,
`no_repeat_ngram_size=6`, with vision/language batch sizes four. Vision models
are released before the language model loads. Positive-label coverage repair
improves agreement with predictions, not their clinical truth.

## Training

`training/head_core.py` and `training/label_mapping.py` expose the original
numerical head-training routines, separated from internal host/audit paths.
`training/lora_core.py` preserves the original LoRA computation. Commands and
the required local manifests are in [TRAINING.md](TRAINING.md).

The visual cohort was **82,608 training studies / 3,483 natural-validation
studies**, patient-disjoint. Scaling and PCA are fitted only on train. The
37 provided pathology categories are mapped approximately to 32 evaluation
categories; eight unsupported categories are masked in loss and set to zero
at deployment. This is not fully supervised training on all 32 diseases.

Heads: 1024–BatchNorm–GELU–512–BatchNorm–GELU–32; AdamW, LR 1e-3,
weight decay 0.01, 10 epochs, batch 512, epoch-level cosine schedule,
positive weights capped at 100. BrainMVP seeds 400–409 select the best
natural-val macro-F1 epoch per seed; FOMO head seeds 600–609 use epoch 10.
No training shuffle is inserted into these frozen head loops.

Language LoRA: Qwen2.5-14B-Instruct, 88,582 supplied-label/report pairs,
2 epochs / 2,770 recorded updates; rank32, alpha64, dropout0.05; seven
attention/MLP projections; LR3e-4, 50 warmup steps, cosine schedule;
per-device batch4, accumulation8, two distributed ranks (effective64);
BF16 with gradient checkpointing; max training length1,024. Loss covers the
serialized conversation, not an independently introduced response-only mask.
Training can include supplied clinical history; deployment uses predicted
labels and `Clinical History: Not provided.`

Earlier documentation's FOMO encoder-adaptation schedule describes an attempted
auxiliary-head training run; it must not be used as evidence of encoder updates.
For actual final representations, use the verified frozen initializer conversion.

## Evaluation, limits and reproducibility boundaries

Previously recorded public **200-study development** readout:

| CRG | Macro-F1 | Mean BLEU | BLEU-4 | ROUGE-L | METEOR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.46956 | 0.20919 | 0.18960 | 0.10560 | 0.22356 | 0.17016 |

These cases were repeatedly inspected during development. They are **not** an
independent test set or the official hidden-test scores. No new scores were
measured by the release audit, and no final hidden-test ranking is inferred.

Release verification checks the retained numerical definitions, syntax,
privacy exclusions, input/checkpoint contracts and weight hashes. A fresh
isolated end-to-end GPU reproduction of this cleaned release has **not** been
completed. Source extraction and path adapters are disclosed in
`SOURCE_PROVENANCE.json`; the Docker archive itself is not claimed bit-identical
to the submitted image. The upstream evaluation dependency and base weights
must be obtained separately. Retraining is not guaranteed bitwise identical:
cohort ordering, dependency versions, Python-set ordering in language-data
preparation and hardware can matter. Use the final trained assets when
reproducing the deployed method, rather than silently retraining a variant.

Key limits: discrete labels discard image detail; mapping is approximate;
eight categories lack supervision; visual false positives/negatives propagate;
three sequences are required; preprocessing can reduce small-lesion visibility;
legacy BrainMVP behavior and train/deployment preprocessing differences remain;
coverage templates do not establish factuality; no external clinical validation.

## References and usage terms

* Rui et al., *Multi-modal Vision Pre-training for Medical Image Analysis*,
  CVPR2025: https://github.com/shaohao011/BrainMVP .
* FOMO2JOMO / MultiModalUNetVAE: https://github.com/jbanusco/fomo25 .
* Qwen: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct .
* Hu et al., *LoRA*: https://arxiv.org/abs/2106.09685 .
* MR-RATE: https://huggingface.co/datasets/Forithmus/MR-RATE .

Identify S-team, submission `1e7ab392`, and the release version when citing this
implementation. Preserve upstream notices. Software licensing is separate from
model/data permissions; see `THIRD_PARTY_NOTICES.md` and the model card.
