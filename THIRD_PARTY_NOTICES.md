# Third-party components and licensing boundaries

The root MIT license applies to S-team's own source and documentation only.
It does not relicense external model weights, datasets, or third-party code.

* `runtime/app/brainmvp/uniformer_blocks.py` is the historical encoder-only
  adaptation of the UniFormer architecture, with the original Facebook
  copyright header retained. UniFormer upstream: SenseTime X-Lab,
  https://github.com/Sense-X/UniFormer (Apache-2.0). The Apache license is
  included in `licenses/Apache-2.0.txt`; changes are marked in the file.
* BrainMVP: Rui et al., CVPR 2025,
  https://github.com/shaohao011/BrainMVP . Obtain the Uniformer checkpoint
  from the authors under their applicable terms. We do not redistribute it
  or assert that the S-team MIT license covers it. Our compatibility wrapper
  deliberately preserves the deployed legacy behavior, not a corrected model.
* FOMO2JOMO MultiModalUNetVAE:
  https://github.com/jbanusco/fomo25 , source commit
  `865ff604e147c4335caba67a71c1e90efdc1a2d2`. The retained upstream
  `pyproject.toml` declares MIT. Obtain the pretrained checkpoint from its
  original release; no original checkpoint is redistributed here. The small
  inference reconstruction in `w045_fomo_runtime.py` is S-team's compatibility
  implementation. See the frozen-encoder correction in README.
* Qwen2.5-14B-Instruct: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct ,
  Apache-2.0. The base model/tokenizer are not redistributed. Language adapters
  are separate from the base model; upstream notices remain applicable.
* Official report-label extraction code is an external dependency, not
  relicensed or vendored in this release. Obtain the matching `evaluation_v2.py`
  through https://github.com/forithmus/VLM3D-Dockers and the challenge resources.
  SHA256: `b4af91a1c0eab7cc5935da5956905576daabe67b226bbac6eeb5537e9a9beb37`.
* MR-RATE: https://huggingface.co/datasets/Forithmus/MR-RATE . Its license,
  access agreement, confidentiality and redistribution restrictions apply
  independently. No MRI, reports, patient metadata or case predictions are
  included. A permissive software license is not permission to commercialize
  the dataset or this trained system.

Dependencies such as PyTorch, NumPy, SciPy, Transformers, PEFT and scikit-learn
retain their own licenses. The minimal local `timm` compatibility namespace is
not a redistribution of the full timm package.
