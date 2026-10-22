# BrainMVP–FOMO Fusion for MRI Report Generation

Method documentation for the **S-team** MR Report Generation submission to
**VLM3D @ MICCAI 2026**.

**Current status: documentation only. Source code and trained weights are not
currently publicly available.** Release plans and access information will be
announced here when confirmed. This repository does not yet provide an
executable implementation or downloadable model.

This page describes the **submitted w045 system**, not later research
prototypes. The submission identifier below was confirmed by the submitting
team; official scores still require the organizer's matching evaluation record.

## Team and submission

**Team:** S-team

| Member | Affiliation |
| --- | --- |
| Xiao Yang | University of Electronic Science and Technology of China, Chengdu, China |
| Jiaqi Gu | Fudan University, Shanghai, China |
| Lei Liu | Fudan University, Shanghai, China |
| Shushang Jiang | Fudan University, Shanghai, China |

Names and affiliations are provided by the submitting team. No corresponding
author designation or author-contribution statement is inferred here.

| Field | Submitted method |
| --- | --- |
| Challenge task | MR Report Generation — VLM3D-MR-REPORT |
| Leaderboard submission ID | `1e7ab392` |
| Method | BrainMVP–FOMO Fusion with Label-Conditioned Report Generation |
| Release shorthand | w045; FOMO probability weight 0.45 |
| Code / trained weights | Not currently publicly available |

## Method overview

Our submitted system separates MRI abnormality prediction from report
generation through a structured label bottleneck.

```text
T1-weighted, T2-weighted, and FLAIR MRI
                    |
        Deterministic series selection
          and 3D image preprocessing
                    |
          +---------+---------+
          |                   |
     BrainMVP C8       FOMO encoder ensemble
          |                   |
     Multilabel heads    Multilabel heads
          +---------+---------+
                    |
       Probability fusion and thresholding
                    |
          Predicted abnormality labels
                    |
       Qwen2.5-14B-Instruct + language LoRA
                    |
     Positive-label coverage check and report
```

- **Visual representations:** pretrained BrainMVP C8 features and an ensemble
  of three FOMO encoder states fine-tuned on challenge training data.
- **Prediction heads:** independently trained multilabel MLP ensembles.
  BrainMVP features are standardized and reduced with PCA; FOMO features are
  standardized separately.
- **Fusion:** probabilities are combined with weights of **0.55 for BrainMVP**
  and **0.45 for FOMO**, followed by a fixed **0.5** threshold.
- **Report generation:** predicted labels condition Qwen2.5-14B-Instruct with
  a trained LoRA adapter (**rank 32, alpha 64**). Inference uses FP16, greedy
  decoding, and a no-repeat-ngram constraint.
- **Coverage checking:** a negation-aware post-generation check appends
  predefined statements for omitted positive conditioning labels when needed.
  This improves consistency with model predictions, not clinical correctness.

The language model receives **predicted labels, not MRI visual tokens**.
The submitted method is therefore not an end-to-end visual-token VLM.
Retrieval-based report rewriting was not deployed.

### Data access and inference inputs

Task-specific training uses the challenge-provided MR-RATE MRI, pathology-label,
report, and split resources. Access to these resources must be obtained through
the challenge/data provider under its applicable terms; this repository does
not redistribute them or provide an alternative download route.

We did not add a separate external raw MRI/report dataset for task-specific
training. BrainMVP, the FOMO25 MultiModalUNetVAE checkpoint, and Qwen are
externally pretrained components; their original pretraining resources and
terms are separate from the challenge training cohort. Using those pretrained
components is not equivalent to distributing their pretraining data here.

The submitted image-only inference route selects T1-weighted, FLAIR, and
T2-weighted MRI. It does not use ground-truth reports, pathology labels,
atlas/segmentation files, or clinical-history text at inference. The language
prompt uses a fixed "Clinical History: Not provided." field followed by the
predicted abnormal findings. SWI is not a model input for this release.

### Submitted preprocessing and visual representations

The following describes the frozen deployment recipe, not a proposed improved
preprocessing pipeline:

1. Inspect each study archive and select one required sequence of each type
   from its image directory. Multiple candidates are resolved deterministically
   by case-folded filename and then full archive-member path, not by labels or
   reference reports. Unsafe archive members are rejected.
2. Read the selected NIfTI arrays as float32 and replace non-finite values with
   zero. Crop to the nonzero foreground bounding box with a 12-voxel margin.
3. Clip intensities at the foreground 0.5th and 99.5th percentiles, normalize
   using foreground mean and standard deviation, clip to [-5, 5], and divide
   by 5. Degenerate standard deviation is handled explicitly.
4. Resize each array to **128 × 128 × 96** using first-order interpolation.
   The three-sequence tensor order is **T1, FLAIR, T2**.
5. Extract the frozen BrainMVP legacy C8 descriptor, **36,864 dimensions**;
   apply training-fitted standardization and PCA to **2,048 dimensions**.
6. For the FOMO branch, use deterministic **64 × 64 × 64** windows covering
   the resized volume, aggregate shared and modality-specific latent means
   across windows and three fine-tuned encoder states, and pool to **1,024
   dimensions**. Apply its separately training-fitted standardization.

The deployment recipe does not introduce an external brain mask or atlas.
Array resizing should not be interpreted as verified physical-coordinate
registration between sequences. The BrainMVP descriptor retains the submitted
legacy layout; results should not be attributed to a subsequently corrected
or redesigned spatial-token pipeline.

## Training and label definitions

The visual-head development cohort comprises **82,608 training studies** and
**3,483 natural-frequency validation studies**, with no patient overlap
between these two cohorts. Feature standardization and PCA are fitted on
training features only. The language adapter is trained using challenge
training reports paired with label-conditioned prompts.

The provided **37-label annotation vocabulary** and the official **32-label
evaluation vocabulary** are different. We use a deterministic mapping to
construct proxy targets in the 32-label space. This mapping includes
approximations and does not create complete supervision for every category.
**Eight categories without corresponding supervision are masked during head
training and assigned zero probability at deployment.**

At inference, the system uses the selected MRI images. Reference reports,
pathology labels, atlas files, and segmentation files are not inference inputs.

### Visual training settings

| Component | Recorded training / selection setting |
| --- | --- |
| BrainMVP representation | Pretrained Uniformer encoder; fixed representation for downstream head training |
| FOMO adaptation | MultiModalUNetVAE initialized from the FOMO2JOMO pretrained checkpoint; seeds 17, 42, 73 |
| FOMO adaptation cohort | 82,608 train studies; patient-disjoint internal fit/stop split of 74,380 / 8,228 studies |
| FOMO adaptation budget | 10 epochs; batch 2; center 64³ crops of the training preprocessing pipeline |
| FOMO learning rates | Encoder 5e-5; auxiliary classification head 1e-3 |
| Selected FOMO epochs | Seed 17: epoch 7; seed 42: epoch 9; seed 73: epoch 9; selected on the internal stop subset |
| Final head architecture | Input → 1024 → BatchNorm → GELU → 512 → BatchNorm → GELU → 32 |
| Final head loss | Masked BCE-with-logits; positive-class weighting capped at 100; eight unsupported classes excluded |
| Final head optimizer | AdamW, learning rate 1e-3, weight decay 0.01 |
| Final head schedule | 10 epochs; batch 512; cosine learning-rate schedule over 10 epochs |
| BrainMVP head ensemble | Ten seeds, 400–409; best natural-validation macro-F1 epoch per seed |
| FOMO head ensemble | Ten seeds, 600–609; fixed epoch 10 |
| PCA | BrainMVP only; 2,048 components; random state 0; fitted on training features only |
| Deployment fusion | Mean sigmoid probabilities within each head ensemble, then 0.55 × BrainMVP + 0.45 × FOMO |
| Deployment decision | Fixed threshold 0.5; unsupported categories forced to probability zero |

The FOMO encoder-adaptation stage uses an auxiliary **37-label** classifier.
The subsequent exported head ensembles use **32 proxy targets**, of which 24
have supervision and eight are masked. The auxiliary head is not the final
deployed classifier. Training-time FOMO center crops and deployment-time
window aggregation are different operations; they are not described as
identical preprocessing.

The eight unsupported deployment categories are `brain_tumor`,
`head_neck_tumor`, `spine_tumor`, `intraventricular_tumor`,
`hydrocephalus_ex_vacuo`, `craniotomy_craniectomy`,
`tumor_resection_cavity`, and `postsurgical_changes`. This concerns these
specific output categories, not a claim that every related disease subtype
has no supervision. Proxy-to-report label bridging also involves
approximations and is not an exact equivalence between the two vocabularies.

### Language adaptation settings

The released-method adapter is `qwen14b_lora_v2` for
**Qwen2.5-14B-Instruct**, not a visual-language version of Qwen. The recorded
training run used **88,582 label/report pairs** from the official training
split and completed **2 epochs / 2,770 optimizer updates**. This is a
report-adaptation cohort, not the 82,608-study visual-head cohort.

Training prompts contain supplied positive pathology labels and, when
available, the provided clinical information; targets are the provided reports.
At deployment, the prompt instead contains predicted labels and no supplied
clinical history. No reference report or ground-truth label is supplied to
the model during inference.

| Setting | Recorded value |
| --- | --- |
| LoRA rank / alpha / dropout | 32 / 64 / 0.05 |
| Adapted projections | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Objective | Causal language-model loss on the serialized training conversation |
| Per-device batch / gradient accumulation | 4 / 8; the training log records two distributed ranks |
| Effective batch | 64 conversation pairs per optimizer update, apart from the final partial batch |
| Learning rate / warmup | 3e-4 / 50 updates |
| Epochs / seed | 2 / 42 |
| Weight decay | 0 |
| Training precision | BF16; gradient checkpointing enabled |
| Tokenization | Qwen chat template; maximum training sequence length 1,024 tokens |
| Inference precision | FP16 |

Values above are based on the saved adapter configuration, recorded training
arguments, training log, and training script. The method uses supervised LoRA
adaptation; it is not reinforcement learning, retrieval training, or joint
MRI-visual-token/language alignment.

### Deployment decoding and reliability

Inference uses greedy decoding (`do_sample=false`), a maximum of **1,024 new
tokens**, and **no_repeat_ngram_size=6**. Tokenized prompts are truncated at
2,048 tokens. The submitted runtime uses vision and language batches of four,
with the visual models released before the language model is loaded.

A negation-aware label check detects positive conditioning findings omitted
from the generated report. Predefined statements are appended for missing
positive findings. This can improve agreement with the predicted labels but
cannot verify their clinical truth or recover false-negative predictions.

Only completed batches are committed to an atomic checkpoint. Resumption
checks the input inventory to avoid mixing studies across runs. Missing
required sequences or preprocessing-invalid inputs follow an explicit
nondiagnostic technical-limitation route; model execution failures are not
silently converted into normal reports.

## Evaluation and results

Visual model selection uses the patient-disjoint **3,483-study natural-frequency
validation cohort** and its **proxy labels**. FOMO encoder checkpoint selection
uses the separate internal training stop subset described above. Proxy-label
validation scores are not interchangeable with report-extracted official
32-label metrics.

### Public 200-study development readout

The canonical w045 report artifact was evaluated on a fixed set of 200 public
studies using the official-compatible report-extraction and scoring pipeline.

| CRG | Macro-F1 | Mean BLEU | BLEU-4 | ROUGE-L | METEOR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.46956 | 0.20919 | 0.18960 | 0.10560 | 0.22356 | 0.17016 |

**These studies were repeatedly inspected during development. These numbers
are a development readout, not independent held-out or hidden-test results.**

### Official hidden-test evaluation

Official hidden-test scores and rankings must be associated with the exact
submission record maintained by the organizers. No verified final ranking or
submission-specific hidden-test score is asserted on this page pending that
record check. Public development scores above must not be substituted for
official hidden-test scores.

The team-confirmed submission identifier is `1e7ab392`. We have not newly
verified a submission-specific final hidden-test score from the organizer's
evaluation record, so no final ranking or hidden-test metric is inferred from
that identifier. Other historical submissions or later experimental variants
must not be conflated with this method.

## Installation, usage, and input/output formats

### What can be used from this repository now

This repository currently provides **documentation only**. Reading it requires
no model installation. There is no public package, executable inference script,
container download, or complete weight download for this method. Consequently,
there is **no public end-to-end installation command** at this time. The
contract below documents the submitted container; it does not claim that
cloning this repository is sufficient to reproduce its results.

### Recorded deployment environment

The submission targets an NVIDIA **A100 80 GB** GPU through a CUDA-enabled
PyTorch container. Vision and language inference run in separate stages. The
container runs without downloading models during inference; all required
assets are supplied separately and must be present before startup.

The frozen container requirement record includes:

| Dependency | Version |
| --- | --- |
| accelerate | 1.14.0 |
| huggingface-hub | 0.36.2 |
| nibabel | 5.4.2 |
| numpy | 2.4.4 |
| peft | 0.20.0 |
| safetensors | 0.8.0 |
| scipy | 1.17.1 |
| transformers | 4.49.0 |

These are the recorded submitted-runtime requirements, not a new environment
validated from this documentation-only repository. A future source release
would also need the exact base container and compatibility checks.

### Input contract

The platform mounts input studies under `/input`, one **`<STUDY>.zip`** per
case. Each archive contains `<STUDY>/img/` with sequence-named NIfTI `.nii.gz`
files. An illustrative, entirely synthetic layout is:

```text
input/
  EXAMPLE_STUDY.zip
    EXAMPLE_STUDY/
      img/
        EXAMPLE_STUDY_t1w-raw-axi.nii.gz
        EXAMPLE_STUDY_flair-raw-axi.nii.gz
        EXAMPLE_STUDY_t2w-raw-axi.nii.gz
```

The ZIP line above denotes the archive's contents, not a directory containing
an unpacked second copy. No actual study identifiers or patient data are
provided in this repository. Atlas and segmentation entries, if supplied by
the platform, are not model inputs.

### Execution contract for the submitted container

Authorized execution requires the original container and all matching model
assets, not just upstream pretrained checkpoints. The platform provides
`/input`, `/weights`, `/output`, and `/checkpoint`. Its container entrypoint
then performs deterministic sequence selection and preprocessing, visual
prediction, label-conditioned decoding, and output/checkpoint writing.

The weight directory must contain the BrainMVP checkpoint, three FOMO states,
feature preprocessors, both head ensembles, label map, Qwen base-model/tokenizer
files, and the trained LoRA adapter listed below. A later source release would
need installation and launch instructions tested with that released artifact
set. No untested public installer is represented here as an available tool.

### Output contract

The submitted runtime writes `/output/predictions.json`. Its schema is:

```json
{
  "generated_reports": [
    {
      "input_image_name": "EXAMPLE_STUDY.nii.gz",
      "report": "Technique: [synthetic example]. Findings: [synthetic example]. Impression: [synthetic example]."
    }
  ]
}
```

`input_image_name` follows the challenge's study identifier convention even
though the physical input is a ZIP. The example is a schema illustration,
not a generated clinical report or an evaluation result. A persistent
checkpoint is maintained separately from the prediction file.

## Submission configuration and final weight inventory

The following artifact inventory corresponds to the team-confirmed w045
submission `1e7ab392`. Filenames are paths **inside the submitted weight
bundle**, not download links or files included in this public repository.

| Component | Bundle-relative file(s) | Availability |
| --- | --- | --- |
| BrainMVP Uniformer | `bmvp/BrainMVP_uniformer.pt` | Not redistributed here; upstream model reference below |
| Fine-tuned FOMO states | `fomo/seed17.pt`, `fomo/seed42.pt`, `fomo/seed73.pt` | Not publicly available |
| BrainMVP normalization/PCA | `bundle/bmvp_preprocessor.npz` | Not publicly available |
| BrainMVP head ensemble | `bundle/bmvp_heads.pt` | Not publicly available |
| FOMO normalization | `bundle/fomo_preprocessor.npz` | Not publicly available |
| FOMO head ensemble | `bundle/fomo_heads.pt` | Not publicly available |
| Deployment label definitions | `bundle/label_map.json` | Not publicly available; masked-category policy documented above |
| Language base | `qwen14b/` model shards, configuration, and tokenizer | Not redistributed here; upstream model card below |
| Language adaptation | `adapter/adapter_config.json`, `adapter/adapter_model.safetensors` | Not publicly available |

Recorded release identifiers, retained for provenance rather than as proof of
public availability:

| Artifact | Recorded SHA-256 |
| --- | --- |
| Final container archive | `fe41ac90c28a0820c6441ee0b17015f69750ef0b46452de6b9b6f2adc953e80c` |
| Final separate weight archive | `89e4312c52cd10e204e654bc5fd399ca25f716ca0794ed209121667a4e72a930` |
| LoRA adapter tensor file | `31169e2ea3a6b776357efbb0e0e87a74f357080273f2442d16313dd2c28b2c50` |

These checksums are transcribed from the retained release/freeze records; the
large archives were not newly rehashed for this documentation update. They
identify recorded artifacts but do not establish clinical effectiveness,
organizer acceptance, or a downloadable weight release.

## Limitations

- The discrete label bottleneck discards spatial, morphological, and
  cross-sequence information that may be important for reporting.
- Label-mapping approximations and eight unsupported categories limit
  coverage; suppressed categories should not be interpreted as learned
  reliable negative predictions.
- Errors in abnormality prediction propagate to the generated report.
- Requiring all three MRI sequences limits applicability to incomplete
  examinations. Image standardization can reduce sensitivity to small findings.
- Positive-label coverage checks cannot recover missed abnormalities or
  establish that a predicted finding is correct.
- The public 200-study readout is not an untouched test set. Independent
  external clinical validation has not been performed.
- Deployment uses shape-based array resizing rather than guaranteed
  inter-sequence physical alignment; the submitted legacy pooling/layout
  should not be mistaken for a validated spatial-localization mechanism.
- Language adaptation uses supplied labels and sometimes clinical history,
  whereas deployment uses predicted labels and no supplied history; this
  training/inference mismatch can affect report reliability.
- Without the original container and exact trained assets, this
  documentation-only repository cannot provide runnable reproduction.

**Research documentation only; not intended for clinical use.**

## Code, weights, and data availability

Source code and trained weights are **not currently publicly available**.
No public download link, access procedure, or release date is confirmed.

This repository contains documentation only. It does not redistribute MRI
images, clinical reports, annotations, checkpoints, upstream pretrained weights,
or private runtime records. Any future release will remain subject to the
applicable data and model licenses.

## References and acknowledgments

This system builds on upstream BrainMVP, FOMO2JOMO/FOMO25 MultiModalUNetVAE,
Qwen2.5, and LoRA. They are not foundation models newly proposed by our team.
In particular, FOMO25 is a challenge name; the specific external representation
used here is the **FOMO2JOMO MultiModalUNetVAE**, not a claim to use every FOMO
entry or a generic official challenge model.

1. Shaohao Rui et al. **Multi-modal Vision Pre-training for Medical Image
   Analysis.** CVPR 2025.
   [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Rui_Multi-modal_Vision_Pre-training_for_Medical_Image_Analysis_CVPR_2025_paper.pdf),
   [author repository](https://github.com/shaohao011/BrainMVP).
2. **FOMO2JOMO: FOMO25 MultiModalUNetVAE implementation and pretrained
   checkpoint.**
   [Author repository](https://github.com/jbanusco/fomo25),
   [v1.0.0 release](https://github.com/jbanusco/fomo25/releases/tag/v1.0.0),
   [FOMO25 challenge](https://fomo25.github.io/).
   Our retained acquisition record identifies source commit
   `865ff604e147c4335caba67a71c1e90efdc1a2d2`; the upstream pretrained
   checkpoint's recorded SHA-256 is
   `2bdd1f19b0759cb300b04ce61808feb7d67ae3c28769331355af7954c406aebf`.
   This upstream checkpoint is distinct from our three fine-tuned states.
3. Qwen Team. **Qwen2.5-14B-Instruct.**
   [Official model card](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct).
4. Edward J. Hu et al. **LoRA: Low-Rank Adaptation of Large Language Models.**
   [Paper](https://arxiv.org/abs/2106.09685).

## Usage terms and release status

This is a public methodological description for the challenge submission.
It is not a clinical product, a software release, or permission to redistribute
challenge data or third-party model weights. Data and upstream components
remain subject to their providers' applicable terms and licenses.

No software or model license is supplied for the unreleased implementation
and trained artifacts. A future release would need its own explicit licensing,
third-party notices, and tested instructions. No release date or private-access
procedure is promised. When referring to the method, please identify S-team,
the challenge task, and submission `1e7ab392`, and acknowledge the upstream
components above.
