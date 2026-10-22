# BrainMVP–FOMO Fusion for MRI Report Generation

Method documentation for the **S-team** MR Report Generation submission to
**VLM3D @ MICCAI 2026**.

**Current status: documentation only. Source code and trained weights are not
currently publicly available.** Release plans and access information will be
announced here when confirmed. This repository does not yet provide an
executable implementation or downloadable model.

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

## Evaluation and results

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

**Research documentation only; not intended for clinical use.**

## Code, weights, and data availability

Source code and trained weights are **not currently publicly available**.
No public download link, access procedure, or release date is confirmed.

This repository contains documentation only. It does not redistribute MRI
images, clinical reports, annotations, checkpoints, upstream pretrained weights,
or private runtime records. Any future release will remain subject to the
applicable data and model licenses.

## Acknowledgments

This system builds on BrainMVP, FOMO, and Qwen2.5. These are upstream model
components, not foundation models newly proposed by our team. Precise citations
and implementation provenance will accompany any implementation release.
