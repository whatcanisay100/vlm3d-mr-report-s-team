
"""Checkpointed w045 phase entrypoint.

The Forithmus phase exposes one unexpanded ``<STUDY>.zip`` per case. This
entrypoint processes those archives lazily in two resumable stages:

1. preprocess -> BrainMVP/FOMO -> frozen w045 label head;
2. release the vision models, then load Qwen2.5-14B+LoRA and decode reports.

Only complete, atomically checkpointed batches are committed. A SIGTERM saves
the last committed state synchronously, allowing a later platform continuation
to resume without treating a model failure as a technical-limitation report.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


APP = Path("/opt/w045/app")
INPUT_ROOT = Path(os.environ.get("W045_INPUT_ROOT", "/input"))
OUTPUT_PATH = Path(os.environ.get("W045_OUTPUT_PATH", "/output/predictions.json"))
CHECKPOINT_PATH = Path(
    os.environ.get("W045_CHECKPOINT_PATH", "/checkpoint/w045_checkpoint.json")
)
SCRATCH_ROOT = Path(os.environ.get("FORITHMUS_SCRATCH", "/tmp"))
WEIGHTS = Path(os.environ.get("FORITHMUS_WEIGHTS_DIR", "/weights"))
VISION_BATCH = 4
LLM_BATCH = 4
SEED = 42


def import_path(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_chain() -> tuple[Any, Any, Any, Any]:
    """Load vendored modules and retarget their frozen asset paths."""
    sys.path.insert(0, str(APP))
    sys.path.insert(0, str(APP / "brainmvp"))
    runtime = import_path(APP / "runtime_io.py", "runtime_io")
    release = import_path(APP / "release_runtime.py", "w045_release_runtime")
    base = import_path(APP / "base_runner.py", "w045_capacity_base")
    recovery = import_path(APP / "recovery_runner.py", "w045_capacity_recovery")

    release.runtime_io = runtime
    base.ROOT = APP
    base.BUNDLE_DIR = WEIGHTS / "bundle"
    base.BMVP_CHECKPOINT = WEIGHTS / "bmvp/BrainMVP_uniformer.pt"
    base.BMVP_SOURCE = APP / "brainmvp/brainmvp_encoder.py"
    base.FOMO_SOURCE = APP / "w045_fomo_runtime.py"
    base.FOMO_STATE_PATHS = {
        seed: WEIGHTS / f"fomo/seed{seed}.pt" for seed in (17, 42, 73)
    }
    base.BASE = WEIGHTS / "qwen14b"
    base.ADAPTER = WEIGHTS / "adapter"
    base.EVALUATOR = APP / "evaluation_v2.py"
    base.TEMPLATE_MODULE = APP / "gen_report.py"
    base.LLM_BATCH = LLM_BATCH
    return runtime, release, base, recovery


def validate_weight_layout(base: Any) -> None:
    required = [
        base.BMVP_CHECKPOINT,
        *(base.FOMO_STATE_PATHS[seed] for seed in (17, 42, 73)),
        base.BUNDLE_DIR / "bmvp_preprocessor.npz",
        base.BUNDLE_DIR / "bmvp_heads.pt",
        base.BUNDLE_DIR / "fomo_preprocessor.npz",
        base.BUNDLE_DIR / "fomo_heads.pt",
        base.BUNDLE_DIR / "label_map.json",
        base.BASE / "config.json",
        base.BASE / "model.safetensors.index.json",
        base.BASE / "tokenizer_config.json",
        base.ADAPTER / "adapter_config.json",
        base.ADAPTER / "adapter_model.safetensors",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required weight files missing: {missing}")


def seed_all(torch: Any) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def initial_state(runtime: Any, release: Any, plan: dict[str, Any]) -> dict[str, Any]:
    state = {
        "schema_version": runtime.SCHEMA_VERSION,
        "candidate_id": release.CANDIDATE_ID,
        "input_fingerprint": plan["input_fingerprint"],
        "case_uids": list(plan["case_uids"]),
        "stage": "vision",
        "next_case_index": 0,
        "probabilities": [],
        "labels": [],
        "reports": [],
        "rng_state": runtime.capture_rng_state(include_torch=True),
    }
    runtime.validate_checkpoint_state(
        state,
        expected_candidate=release.CANDIDATE_ID,
        expected_input_fingerprint=plan["input_fingerprint"],
    )
    return state


def save_state(runtime: Any, store: Any, state: dict[str, Any]) -> None:
    state["rng_state"] = runtime.capture_rng_state(include_torch=True)
    store.save(state)


def preprocess_case(runtime: Any, case: Any, scratch: Path) -> np.ndarray:
    selection = runtime.scan_case_zip(case)
    extracted = runtime.extract_selected_series(case, selection, scratch)
    row = []
    for modality in runtime.REQUIRED_MODALITIES:
        array = runtime.preprocess_original_img(extracted[modality])
        if (
            array.shape != (128, 128, 96)
            or array.dtype != np.float32
            or not np.isfinite(array).all()
            or not np.any(array)
        ):
            raise RuntimeError(f"preprocessed volume invalid for {case.uid}/{modality}")
        row.append(array)
    return np.stack(row).astype(np.float32, copy=False)


def load_vision_models(base: Any, device: Any) -> tuple[Any, Any, list[Any]]:
    import torch

    from brainmvp_encoder import BrainMVPEncoder

    encoder = BrainMVPEncoder(str(base.BMVP_CHECKPOINT)).to(device).eval()
    architecture = import_path(base.FOMO_SOURCE, "w045_release_fomo_arch")
    models = []
    for seed in (17, 42, 73):
        payload = torch.load(
            base.FOMO_STATE_PATHS[seed], map_location="cpu", weights_only=True
        )
        if payload.get("seeds") != [seed] or len(payload.get("models", [])) != 1:
            raise RuntimeError(f"FOMO state drift: seed={seed}")
        model = architecture.FrozenFomoRepresentation()
        model.load_state_dict(payload["models"][0], strict=True)
        models.append(model.to(device).eval())
    return encoder, architecture, models


def extract_bmvp(base: Any, encoder: Any, volumes: np.ndarray, device: Any) -> np.ndarray:
    import torch

    pieces = []
    with torch.inference_mode():
        for start in range(0, len(volumes), VISION_BATCH):
            stop = min(start + VISION_BATCH, len(volumes))
            modality_tokens = []
            for modality in range(3):
                values = torch.from_numpy(volumes[start:stop, modality, None]).to(device)
                *_, feature_map = encoder.uniformer(values)
                batch, channels, depth, height, width = feature_map.shape
                if (channels, depth, height, width) != (512, 22, 8, 8):
                    raise RuntimeError(f"BrainMVP shape drift: {feature_map.shape}")
                modality_tokens.append(
                    feature_map.reshape(batch, depth * height * width, channels)
                )
            tokens = torch.cat(modality_tokens, dim=1).float().cpu().numpy()
            pieces.append(base.c8_pool(tokens))
    output = np.concatenate(pieces).astype(np.float32)
    if output.shape != (len(volumes), 36_864) or not np.isfinite(output).all():
        raise RuntimeError(f"BrainMVP output drift: {output.shape}")
    return output


def extract_fomo(
    architecture: Any, models: list[Any], volumes: np.ndarray, device: Any
) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    starts = [
        np.linspace(0, size - 64, int(np.ceil(size / 64)))
        .round()
        .astype(int)
        .tolist()
        for size in (128, 128, 96)
    ]
    pieces = []
    with torch.inference_mode():
        for row_index in range(len(volumes)):
            per_modality = []
            for modality in range(3):
                volume = torch.from_numpy(volumes[row_index, modality][None, None]).to(
                    device
                )
                shared_acc, specific_acc = [], []
                for depth in starts[0]:
                    for height in starts[1]:
                        for width in starts[2]:
                            patch = volume[
                                :, :, depth : depth + 64, height : height + 64, width : width + 64
                            ]
                            shared_sum = None
                            specific_parts = []
                            for model in models:
                                shared, specific = model.encode_modality(patch)
                                shared_sum = (
                                    shared / len(models)
                                    if shared_sum is None
                                    else shared_sum + shared / len(models)
                                )
                                specific_parts.append(specific / len(models))
                            shared_acc.append(shared_sum)
                            specific_acc.append(torch.stack(specific_parts).sum(0))
                per_modality.append(
                    (torch.stack(shared_acc).mean(0), torch.stack(specific_acc).mean(0))
                )
            latent = torch.cat(
                [
                    torch.stack([item[0] for item in per_modality]).mean(0),
                    *[item[1] for item in per_modality],
                ],
                dim=1,
            )
            pieces.append(
                functional.adaptive_avg_pool3d(latent, 1)
                .flatten(1)
                .float()
                .cpu()
                .numpy()
            )
    output = np.concatenate(pieces).astype(np.float32)
    if output.shape != (len(volumes), 1_024) or not np.isfinite(output).all():
        raise RuntimeError(f"FOMO output drift: {output.shape}")
    return output


def run_vision_stage(
    runtime: Any,
    release: Any,
    base: Any,
    plan: dict[str, Any],
    cases: dict[str, Any],
    state: dict[str, Any],
    store: Any,
    device: Any,
) -> None:
    import torch

    encoder, architecture, models = load_vision_models(base, device)
    runtime.restore_rng_state(state["rng_state"], include_torch=True)
    started = time.time()
    try:
        while state["next_case_index"] < len(plan["records"]):
            begin = state["next_case_index"]
            end = min(begin + VISION_BATCH, len(plan["records"]))
            routes = plan["records"][begin:end]
            batch_values: dict[str, tuple[list[float], list[str]]] = {}
            volumes: list[np.ndarray] = []
            candidate_uids: list[str] = []
            scratch = Path(
                tempfile.mkdtemp(prefix=f"w045-{os.getpid()}-", dir=SCRATCH_ROOT)
            )
            try:
                for route in routes:
                    if route["route"] != "candidate":
                        continue
                    uid = route["uid"]
                    try:
                        volumes.append(preprocess_case(runtime, cases[uid], scratch))
                        candidate_uids.append(uid)
                    except Exception as error:
                        print(
                            f"[w045] {uid} preprocessing unusable: {type(error).__name__}",
                            flush=True,
                        )
                if volumes:
                    stacked = np.stack(volumes).astype(np.float32)
                    bmvp = extract_bmvp(base, encoder, stacked, device)
                    fomo = extract_fomo(architecture, models, stacked, device)
                    base.COHORT_UIDS = list(candidate_uids)
                    scores = base.score_candidate(bmvp, fomo)
                    for index, uid in enumerate(candidate_uids):
                        batch_values[uid] = (
                            scores["fused"][index].astype(float).tolist(),
                            list(scores["report_labels"][index]),
                        )
            finally:
                shutil.rmtree(scratch, ignore_errors=True)

            probabilities = []
            labels = []
            for route in routes:
                value = batch_values.get(route["uid"])
                probabilities.append(None if value is None else value[0])
                labels.append([] if value is None else value[1])
            state["probabilities"].extend(probabilities)
            state["labels"].extend(labels)
            state["next_case_index"] = end
            save_state(runtime, store, state)
            print(
                f"[w045] vision {end}/{len(plan['records'])} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    finally:
        del encoder
        models.clear()
        gc.collect()
        torch.cuda.empty_cache()

    state["stage"] = "report"
    state["next_case_index"] = 0
    save_state(runtime, store, state)


def load_report_model(base: Any, device: Any) -> tuple[Any, Any, Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    evaluator, templates, _ = base.load_and_validate_coverage(lightweight=True)
    capacity = base.audit_llm_load_capacity(torch, device)
    print(json.dumps({"llm_load_capacity": capacity}, sort_keys=True), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        base.BASE, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base.BASE,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map={"": device},
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(
        model,
        base.ADAPTER,
        local_files_only=True,
        torch_device=str(device),
        ephemeral_gpu_offload=False,
    )
    model.eval()
    base.audit_model_residency(model, device)
    return tokenizer, model, evaluator, templates


def run_report_stage(
    runtime: Any,
    release: Any,
    base: Any,
    recovery: Any,
    plan: dict[str, Any],
    state: dict[str, Any],
    store: Any,
    device: Any,
) -> None:
    import torch

    tokenizer, model, evaluator, templates = load_report_model(base, device)
    runtime.restore_rng_state(state["rng_state"], include_torch=True)
    started = time.time()
    try:
        while state["next_case_index"] < len(plan["records"]):
            begin = state["next_case_index"]
            end = min(begin + LLM_BATCH, len(plan["records"]))
            rows = [
                {
                    "uid": plan["records"][index]["uid"],
                    "report_labels": state["labels"][index],
                    "group": "phase",
                }
                for index in range(begin, end)
                if state["probabilities"][index] is not None
            ]
            decoded_by_uid: dict[str, dict[str, Any]] = {}
            if rows:
                with open("/tmp/decode_records.jsonl", "wb") as handle:
                    decoded = recovery.decode_rows(
                        base,
                        torch,
                        tokenizer,
                        model,
                        evaluator,
                        templates,
                        rows,
                        "candidate_greedy_no_repeat_ngram6",
                        6,
                        handle,
                    )
                if any(not record.get("valid") for record in decoded):
                    raise RuntimeError("generated candidate report failed frozen validity gate")
                decoded_by_uid = {record["uid"]: record for record in decoded}

            reports = []
            for index in range(begin, end):
                uid = plan["records"][index]["uid"]
                if state["probabilities"][index] is None:
                    report = release.TECHNICAL_LIMITATION_REPORT
                else:
                    report = decoded_by_uid[uid]["report"]
                reports.append(
                    {
                        "uid": uid,
                        "input_image_name": plan["records"][index]["input_image_name"],
                        "report": report,
                    }
                )
            state["reports"].extend(reports)
            state["next_case_index"] = end
            save_state(runtime, store, state)
            print(
                f"[w045] report {end}/{len(plan['records'])} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    state["stage"] = "complete"
    save_state(runtime, store, state)


def main() -> None:
    import torch

    global INPUT_ROOT, OUTPUT_PATH, CHECKPOINT_PATH
    
    
    INPUT_ROOT = INPUT_ROOT.resolve()
    OUTPUT_PATH = OUTPUT_PATH.resolve()
    CHECKPOINT_PATH = CHECKPOINT_PATH.resolve()

    
    
    try:
        os.chmod(OUTPUT_PATH.parent, 0o777)
    except OSError:
        pass

    
    contract_only = os.environ.get("W045_CONTRACT_TEST") == "1" and not torch.cuda.is_available()
    if torch.cuda.is_available() and torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device is required")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    seed_all(torch)

    runtime, release, base, recovery = load_chain()
    if contract_only:
        
        
        uids = sorted(p.stem for p in INPUT_ROOT.glob("*.zip"))
        if not uids:
            raise RuntimeError("no study zips in /input")
        state = {
            "schema_version": runtime.SCHEMA_VERSION,
            "candidate_id": release.CANDIDATE_ID,
            "input_fingerprint": "0" * 64,  
            "case_uids": uids,
            "stage": "complete",
            "next_case_index": len(uids),
            "probabilities": [[] for _ in uids],
            "labels": [[] for _ in uids],
            "reports": [
                {"uid": uid, "input_image_name": f"{uid}.nii.gz",
                 "report": "Contract-only placeholder (no CUDA in local test)."}
                for uid in uids
            ],
            "rng_state": runtime.capture_rng_state(include_torch=True),
        }
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        runtime.write_predictions(OUTPUT_PATH, state)
        print(f"[w045] contract-only DONE {len(uids)} reports -> {OUTPUT_PATH}", flush=True)
        return

    validate_weight_layout(base)
    plan = release.build_route_plan(INPUT_ROOT)
    cases = {case.uid: case for case in runtime.discover_cases(INPUT_ROOT)}
    if list(cases) != plan["case_uids"]:
        raise RuntimeError("live case UID order differs from route plan")

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    store = runtime.AtomicCheckpointStore(
        CHECKPOINT_PATH,
        expected_candidate=release.CANDIDATE_ID,
        expected_input_fingerprint=plan["input_fingerprint"],
    )
    state = store.load() if CHECKPOINT_PATH.exists() else initial_state(runtime, release, plan)
    runtime.install_sigterm_checkpoint(lambda: save_state(runtime, store, state))

    if state["stage"] == "vision":
        run_vision_stage(runtime, release, base, plan, cases, state, store, device)
    if state["stage"] == "report":
        run_report_stage(runtime, release, base, recovery, plan, state, store, device)
    if state["stage"] != "complete":
        raise RuntimeError(f"unexpected terminal stage: {state['stage']}")
    runtime.write_predictions(OUTPUT_PATH, state)
    print(
        f"[w045] DONE {len(state['reports'])} reports -> {OUTPUT_PATH}", flush=True
    )


if __name__ == "__main__":
    main()
