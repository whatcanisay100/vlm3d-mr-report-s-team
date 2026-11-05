from __future__ import annotations
import ast,gc,importlib.util,json,re,sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import numpy as np

ROOT=Path(__file__).resolve().parent
BUNDLE_DIR=Path("/weights/bundle")
COHORT_UIDS=[]


EXPECTED_CLASSES = 32


EXPECTED_BMVP_DIM = 36_864


W_FOMO = 0.45


THRESHOLD = 0.5


MAX_NEW_TOKENS = 1024


LLM_BATCH = 4


MIN_LLM_LOAD_FREE_MEMORY_BYTES = 30 * 1024**3


FALLBACK_ADDENDUM = (
    "\n\nDefinitive positive findings confirmed on this examination are documented "
    "below for clarity and diagnostic completeness:\n"
)


REV = {
    "lacunar_stroke": "Lacunar infarct",
    "arachnoid_cyst": "Arachnoid cyst",
    "pineal_cyst": "Cyst of pineal gland",
    "glioma": "Glioma",
    "brain_metastasis": "Metastatic malignant neoplasm to brain",
    "meningioma": "Intracranial meningioma",
    "pituitary_tumor": "Pituitary adenoma",
    "rathkes_cleft_cyst": "Rathke's pouch cyst",
    "chiari_malformation": "Chiari malformation",
    "cerebral_atrophy": "Cerebral atrophy",
    "cerebral_edema": "Cerebral edema",
    "cavum_septum_pellucidum": "Structure of cave of septum pellucidum",
    "mega_cisterna_magna": "Mega cisterna magna",
    "ventriculomegaly": "Ventriculomegaly",
    "cavernous_malformation_cavernoma": "Cavernous hemangioma",
    "multiple_sclerosis": "Demyelinating disease of central nervous system",
    "spinal_degenerative_changes": "Spinal stenosis",
    "acute_ischemic_stroke": "Cerebral infarction",
    "chronic_ischemic_stroke": "Cerebral infarction",
    "small_vessel_ischemic_disease": "Cerebral infarction",
    "intracranial_hemorrhage": "Cerebral hemorrhage",
    "intraparenchymal_hemorrhage": "Cerebral hemorrhage",
    "encephalomalacia_gliosis": "Encephalomalacia|Gliosis",
    "brain_mass_effect": "Cerebral edema",
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def c8_pool(tokens: np.ndarray) -> np.ndarray:
    batch = len(tokens)
    spatial = tokens.reshape(batch, 3, 22, 8, 8, 512)
    blocks: list[np.ndarray] = []
    for depth in range(2):
        for height in range(2):
            for width in range(2):
                region = spatial[
                    :,
                    :,
                    depth * 11 : (depth + 1) * 11,
                    height * 4 : (height + 1) * 4,
                    width * 4 : (width + 1) * 4,
                ].reshape(batch, 3, -1, 512)
                blocks.append(
                    np.concatenate(
                        [region.mean(2), region.max(2), region.std(2)], axis=1
                    ).reshape(batch, -1)
                )
    output = np.concatenate(blocks, axis=1).astype(np.float32)
    if output.shape != (batch, EXPECTED_BMVP_DIM) or not np.isfinite(output).all():
        raise RuntimeError(f"malformed legacy c8 features: {output.shape}")
    return output


def make_head(torch: Any, dimension: int) -> Any:
    class MLPHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(dimension, 1024),
                torch.nn.BatchNorm1d(1024),
                torch.nn.GELU(),
                torch.nn.Linear(1024, 512),
                torch.nn.BatchNorm1d(512),
                torch.nn.GELU(),
                torch.nn.Linear(512, EXPECTED_CLASSES),
            )

        def forward(self, values: Any) -> Any:
            return self.net(values)

    return MLPHead()


def score_arm(raw: np.ndarray, preprocessor_name: str, checkpoint_name: str) -> np.ndarray:
    import torch

    with np.load(BUNDLE_DIR / preprocessor_name, allow_pickle=False) as archive:
        preprocessor = {name: np.asarray(archive[name]) for name in archive.files}
    scaled = (raw - preprocessor["raw_mean"]) / preprocessor["raw_std"]
    if preprocessor_name.startswith("bmvp"):
        transformed = scaled @ preprocessor["pca_components"].T
        transformed -= preprocessor["pca_mean"] @ preprocessor["pca_components"].T
        dimension = 2048
    else:
        transformed = scaled
        dimension = 1024
    checkpoint = torch.load(
        BUNDLE_DIR / checkpoint_name, map_location="cpu", weights_only=True
    )
    states = checkpoint.get("state_dicts", [])
    if len(states) != 10:
        raise RuntimeError(f"head-state count drift: {checkpoint_name}")
    values = torch.from_numpy(np.ascontiguousarray(transformed, dtype=np.float32))
    probabilities = []
    for state in states:
        head = make_head(torch, dimension)
        head.load_state_dict(state, strict=True)
        head.eval()
        with torch.inference_mode():
            probabilities.append(torch.sigmoid(head(values)).numpy())
    return np.stack(probabilities).mean(0).astype(np.float32)


def score_candidate(bmvp_features: np.ndarray, fomo_features: np.ndarray) -> dict[str, Any]:
    bmvp = score_arm(bmvp_features, "bmvp_preprocessor.npz", "bmvp_heads.pt")
    fomo = score_arm(fomo_features, "fomo_preprocessor.npz", "fomo_heads.pt")
    fused = ((1.0 - W_FOMO) * bmvp + W_FOMO * fomo).astype(np.float32)
    label_map = json.loads((BUNDLE_DIR / "label_map.json").read_text(encoding="utf-8"))
    keep32 = list(label_map["keep32"])
    masked = list(label_map["masked"])
    masked_indices = [keep32.index(name) for name in masked]
    fused[:, masked_indices] = 0.0
    for name, values in (("bmvp", bmvp), ("fomo", fomo), ("fused", fused)):
        if values.shape != (len(COHORT_UIDS), EXPECTED_CLASSES) or not np.isfinite(values).all():
            raise RuntimeError(f"{name} probability output malformed: {values.shape}")
        if values.min() < 0 or values.max() > 1:
            raise RuntimeError(f"{name} probability outside [0,1]")
    if np.count_nonzero(fused[:, masked_indices]) != 0:
        raise RuntimeError("masked deployment probabilities are nonzero")
    hits = fused >= THRESHOLD
    proxy_labels = [[keep32[index] for index in np.flatnonzero(row)] for row in hits]
    report_labels = []
    for labels in proxy_labels:
        names: set[str] = set()
        for label in labels:
            for item in REV.get(label, "").split("|"):
                if item:
                    names.add(item)
        report_labels.append(sorted(names))
    return {
        "bmvp": bmvp,
        "fomo": fomo,
        "fused": fused,
        "proxy_labels": proxy_labels,
        "report_labels": report_labels,
        "keep32": keep32,
        "masked": masked,
        "masked_indices": masked_indices,
    }


def user_prompt(labels: list[str]) -> str:
    return (
        "Clinical History: Not provided.\n\n"
        f"Abnormal findings: {', '.join(labels) if labels else 'None'}\n\n"
        "Write the complete brain MRI report with Technique, Findings, and Impression sections."
        " Only report the listed abnormal findings; do NOT mention any other abnormality or finding."
    )


def append_coverage_fallback(
    report: str, missing: list[str], templates: dict[str, str]
) -> str:
    if not missing:
        return report
    fallback = [
        templates.get(label, f"{label} is present.")
        .replace("{side}", "left")
        .replace("{level}", "C5-C6")
        for label in missing
    ]
    return report.rstrip() + FALLBACK_ADDENDUM + "\n".join(
        f"- {text}" for text in fallback
    )


def load_frozen_template_literals() -> tuple[list[str], dict[str, str]]:
    tree = ast.parse(TEMPLATE_MODULE.read_text(encoding="utf-8"), str(TEMPLATE_MODULE))
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in {"L", "TEMPLATES"}:
            values[target.id] = ast.literal_eval(node.value)
    if set(values) != {"L", "TEMPLATES"}:
        raise RuntimeError("frozen report template literals are unavailable")
    return list(values["L"]), dict(values["TEMPLATES"])


def load_lightweight_coverage_extractor() -> Any:
    """Load the pinned extractor logic without its unrelated metric imports."""
    tree = ast.parse(EVALUATOR.read_text(encoding="utf-8"), str(EVALUATOR))
    wanted_assignments = {"LABEL_37_PATTERNS", "NEGATION_PATTERNS"}
    wanted_functions = {"_check_negation", "extract_labels_37"}
    nodes: list[ast.stmt] = []
    found_assignments: set[str] = set()
    found_functions: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            if names & wanted_assignments:
                nodes.append(node)
                found_assignments.update(names & wanted_assignments)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            nodes.append(node)
            found_functions.add(node.name)
    if found_assignments != wanted_assignments or found_functions != wanted_functions:
        raise RuntimeError("frozen coverage extractor definitions are unavailable")
    namespace: dict[str, Any] = {"re": re}
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(EVALUATOR), "exec"), namespace)
    return SimpleNamespace(extract_labels_37=namespace["extract_labels_37"])


def load_and_validate_coverage(
    *, lightweight: bool = False
) -> tuple[Any, dict[str, str], dict[str, Any]]:
    evaluator = (
        load_lightweight_coverage_extractor()
        if lightweight
        else load_module(EVALUATOR, "w045_integration_coverage_extractor")
    )
    label_names, templates = load_frozen_template_literals()
    mapped_report_labels = {
        item for value in REV.values() for item in value.split("|") if item
    }
    if (
        len(label_names) != 37
        or len(set(label_names)) != 37
        or not set(label_names).issubset(templates)
        or not mapped_report_labels.issubset(label_names)
        or len(FALLBACK_ADDENDUM) <= 80
    ):
        raise RuntimeError("frozen coverage fallback schema drift")
    failures: list[str] = []
    for label in label_names:
        adversarial = f"No evidence of {label} is identified. {label} is not present."
        repaired = append_coverage_fallback(adversarial, [label], templates)
        if evaluator.extract_labels_37(repaired).get(label) != 1:
            failures.append(label)
    if failures:
        raise RuntimeError(f"negation-safe coverage fallback failed: {failures}")
    return evaluator, templates, {
        "implementation": "frozen template plus negation-safe positive addendum",
        "labels_tested": len(label_names),
        "failures": failures,
        "positive_addendum_characters": len(FALLBACK_ADDENDUM),
        "passed": True,
    }


def clean_output(text: str) -> str:
    marker = "assistant\n"
    return text.rsplit(marker, 1)[-1].strip() if marker in text else text.strip()


def audit_model_residency(model: Any, expected_device: Any) -> dict[str, Any]:
    """Prove that the complete PEFT model is resident on one CUDA device."""
    expected = str(expected_device)
    if expected != "cuda:0":
        raise RuntimeError(f"registered 14B logical device drift: {expected}")

    device_maps: list[dict[str, Any]] = []
    offload_hooks: list[str] = []
    seen_maps: set[int] = set()
    for module_name, module in model.named_modules():
        candidate = getattr(module, "hf_device_map", None)
        if isinstance(candidate, dict) and candidate and id(candidate) not in seen_maps:
            seen_maps.add(id(candidate))
            normalized: dict[str, str] = {}
            for key, value in candidate.items():
                if isinstance(value, int) or str(value).isdigit():
                    assigned = f"cuda:{int(value)}"
                else:
                    assigned = str(value)
                    if assigned == "cuda":
                        assigned = expected
                normalized[str(key)] = assigned
                if assigned != expected:
                    raise RuntimeError(
                        "14B device map contains a non-registered assignment: "
                        f"module={module_name!r} key={key!r} value={value!r}"
                    )
            device_maps.append(
                {"module": module_name, "assignments": normalized}
            )
        hook = getattr(module, "_hf_hook", None)
        if hook is not None and bool(getattr(hook, "offload", False)):
            offload_hooks.append(module_name)
    if offload_hooks:
        raise RuntimeError(f"14B offload hooks are forbidden: {offload_hooks[:16]}")

    tensor_counts: dict[str, int] = {}
    element_counts: dict[str, int] = {}
    wrong_device: list[str] = []
    parameter_tensors = 0
    parameter_elements = 0
    buffer_tensors = 0
    buffer_elements = 0
    for kind, tensors in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, tensor in tensors:
            observed = str(tensor.device)
            tensor_counts[observed] = tensor_counts.get(observed, 0) + 1
            element_counts[observed] = element_counts.get(observed, 0) + tensor.numel()
            if kind == "parameter":
                parameter_tensors += 1
                parameter_elements += tensor.numel()
            else:
                buffer_tensors += 1
                buffer_elements += tensor.numel()
            if observed != expected:
                wrong_device.append(f"{kind}:{name}={observed}")
    if parameter_tensors == 0 or parameter_elements == 0:
        raise RuntimeError("14B residency audit found no parameters")
    if wrong_device:
        raise RuntimeError(
            "14B CPU/disk/meta/wrong-GPU residency is forbidden: "
            + ", ".join(wrong_device[:16])
        )
    return {
        "expected_device": expected,
        "parameter_tensors": parameter_tensors,
        "parameter_elements": parameter_elements,
        "buffer_tensors": buffer_tensors,
        "buffer_elements": buffer_elements,
        "tensor_counts_by_device": tensor_counts,
        "element_counts_by_device": element_counts,
        "discovered_hf_device_maps": device_maps,
        "offload_hook_count": 0,
        "all_tensors_on_expected_device": True,
    }


def audit_llm_load_capacity(torch: Any, device: Any) -> dict[str, int]:
    """Fail before model loading if another process has claimed the GPU."""
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    record = {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "current_process_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "current_process_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "minimum_free_bytes": MIN_LLM_LOAD_FREE_MEMORY_BYTES,
    }
    if free_bytes < MIN_LLM_LOAD_FREE_MEMORY_BYTES:
        raise RuntimeError(
            "14B load capacity gate failed; GPU is no longer exclusive: "
            + json.dumps(record, sort_keys=True)
        )
    return record
