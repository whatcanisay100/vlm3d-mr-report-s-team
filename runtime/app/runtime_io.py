
"""Fail-closed phase I/O, preprocessing, and checkpoint primitives.

This module deliberately contains no model loading or submission code.  It is
the versioned boundary between Forithmus' flat ZIP input schema and the frozen
w045 model pipeline.  The legacy ``docker/`` placeholder remains untouched.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
import signal
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping


SCHEMA_VERSION = "w045_checkpoint_v1"
CANDIDATE_ID = "w045_14b_fp16_frozen"
UID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
MODALITIES = ("t1", "flair", "t2", "swi")
REQUIRED_MODALITIES = ("t1", "flair", "t2")


class RuntimeContractError(RuntimeError):
    """Raised when phase input or checkpoint state violates the frozen contract."""


@dataclass(frozen=True)
class CaseSpec:
    uid: str
    archive: Path
    input_image_name: str


@dataclass(frozen=True)
class SeriesSelection:
    uid: str
    selected: Mapping[str, str]
    candidates: Mapping[str, tuple[str, ...]]
    ignored_atlas_members: int
    ignored_seg_members: int
    member_count: int
    uncompressed_bytes: int


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink():
        raise RuntimeContractError(f"Refusing symlink output directory: {target.parent}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", suffix=".tmp", dir=target.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(payload))


def _require_regular_no_symlink(path: Path, label: str) -> os.stat_result:
    if path.is_symlink() or not path.is_file():
        raise RuntimeContractError(f"{label} must be a regular, non-symlink file: {path}")
    return path.stat()


def discover_cases(input_root: str | Path) -> list[CaseSpec]:
    """Discover a stable UID order from the phase's flat ``*.zip`` input."""

    root = Path(input_root)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeContractError(f"Input root must be a non-symlink directory: {root}")
    archives = sorted(root.glob("*.zip"), key=lambda path: path.name.encode("utf-8"))
    if not archives:
        raise RuntimeContractError(f"No phase ZIP inputs found under {root}")
    cases: list[CaseSpec] = []
    seen: set[str] = set()
    for archive in archives:
        _require_regular_no_symlink(archive, "Case archive")
        uid = archive.name[:-4]
        if not UID_PATTERN.fullmatch(uid):
            raise RuntimeContractError(f"Unsafe or unsupported study UID: {uid!r}")
        if uid in seen:
            raise RuntimeContractError(f"Duplicate study UID: {uid}")
        seen.add(uid)
        cases.append(CaseSpec(uid=uid, archive=archive, input_image_name=f"{uid}.nii.gz"))
    return cases


def input_inventory_fingerprint(cases: Iterable[CaseSpec]) -> str:
    records = []
    for case in cases:
        info = _require_regular_no_symlink(case.archive, "Case archive")
        records.append(
            {"uid": case.uid, "archive_name": case.archive.name, "archive_bytes": info.st_size}
        )
    return hashlib.sha256(canonical_json_bytes(records)).hexdigest()


def _safe_zip_path(name: str, uid: str) -> PurePosixPath:
    if not name or "\\" in name or name.startswith("/"):
        raise RuntimeContractError(f"Unsafe ZIP member path: {name!r}")
    member = PurePosixPath(name)
    if any(part in {"", ".", ".."} for part in member.parts):
        raise RuntimeContractError(f"Unsafe ZIP member path: {name!r}")
    if not member.parts or member.parts[0] != uid:
        raise RuntimeContractError(
            f"ZIP member is outside the declared study directory {uid!r}: {name!r}"
        )
    return member


def _classify_series(filename: str) -> str | None:
    lower = filename.casefold()
    if not lower.endswith(".nii.gz") or "mask" in lower:
        return None
    stem = lower[:-7]
    tokens = {token for token in re.split(r"[^a-z0-9]+", stem) if token}
    if "flair" in tokens:
        return "flair"
    if "swi" in tokens:
        return "swi"
    if ({"t1", "t1w"} & tokens) and not ({"t1ce", "t1c"} & tokens):
        return "t1"
    if {"t2", "t2w"} & tokens:
        return "t2"
    return None


def scan_case_zip(
    case: CaseSpec,
    *,
    max_members: int = 512,
    max_member_bytes: int = 2 * 1024**3,
    max_total_bytes: int = 16 * 1024**3,
) -> SeriesSelection:
    """Validate a case ZIP and choose raw-image series deterministically.

    Selection is independent of central-directory order: candidates are sorted
    by UTF-8 basename and then by full member path.  Atlas and segmentation
    members are counted but never selected for the frozen w045 input route.
    """

    _require_regular_no_symlink(case.archive, "Case archive")
    candidates: dict[str, list[str]] = {key: [] for key in MODALITIES}
    ignored_atlas = 0
    ignored_seg = 0
    total_bytes = 0
    seen: set[str] = set()
    with zipfile.ZipFile(case.archive, "r") as archive:
        infos = archive.infolist()
        if len(infos) > max_members:
            raise RuntimeContractError(
                f"ZIP member count exceeds limit for {case.uid}: {len(infos)} > {max_members}"
            )
        for info in infos:
            member = _safe_zip_path(info.filename, case.uid)
            if info.filename in seen:
                raise RuntimeContractError(f"Duplicate ZIP member: {info.filename}")
            seen.add(info.filename)
            if info.flag_bits & 0x1:
                raise RuntimeContractError(f"Encrypted ZIP member is not supported: {info.filename}")
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if unix_mode and stat.S_ISLNK(unix_mode):
                raise RuntimeContractError(f"Symlink ZIP member is forbidden: {info.filename}")
            if info.file_size > max_member_bytes:
                raise RuntimeContractError(f"ZIP member exceeds size limit: {info.filename}")
            total_bytes += info.file_size
            if total_bytes > max_total_bytes:
                raise RuntimeContractError(f"ZIP expands beyond size limit for {case.uid}")
            if info.is_dir():
                continue
            if len(member.parts) >= 2 and member.parts[1] == "atlas":
                ignored_atlas += 1
                continue
            if len(member.parts) >= 2 and member.parts[1] == "seg":
                ignored_seg += 1
                continue
            if len(member.parts) != 3 or member.parts[1] != "img":
                raise RuntimeContractError(f"Unexpected case member layout: {info.filename}")
            modality = _classify_series(member.name)
            if modality is not None:
                candidates[modality].append(info.filename)
    ordered = {
        modality: tuple(
            sorted(paths, key=lambda value: (PurePosixPath(value).name.casefold(), value))
        )
        for modality, paths in candidates.items()
    }
    selected = {modality: paths[0] for modality, paths in ordered.items() if paths}
    return SeriesSelection(
        uid=case.uid,
        selected=selected,
        candidates=ordered,
        ignored_atlas_members=ignored_atlas,
        ignored_seg_members=ignored_seg,
        member_count=len(seen),
        uncompressed_bytes=total_bytes,
    )


def extract_selected_series(
    case: CaseSpec, selection: SeriesSelection, scratch_root: str | Path
) -> dict[str, Path]:
    """Extract only selected raw series to per-case writable scratch."""

    if case.uid != selection.uid:
        raise RuntimeContractError("Case and series-selection UIDs differ")
    scratch = Path(scratch_root)
    scratch.mkdir(parents=True, exist_ok=True)
    if scratch.is_symlink():
        raise RuntimeContractError(f"Scratch root may not be a symlink: {scratch}")
    case_dir = scratch / case.uid
    if case_dir.exists():
        raise RuntimeContractError(f"Refusing existing case scratch directory: {case_dir}")
    case_dir.mkdir(mode=0o700)
    extracted: dict[str, Path] = {}
    try:
        with zipfile.ZipFile(case.archive, "r") as archive:
            for modality, member_name in selection.selected.items():
                basename = PurePosixPath(member_name).name
                destination = case_dir / basename
                temporary = destination.with_name(f".{destination.name}.partial")
                with archive.open(member_name, "r") as source, temporary.open("xb") as sink:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        sink.write(chunk)
                    sink.flush()
                    os.fsync(sink.fileno())
                os.replace(temporary, destination)
                extracted[modality] = destination
    except BaseException:
        for child in case_dir.iterdir():
            if child.is_file() and not child.is_symlink():
                child.unlink()
        case_dir.rmdir()
        raise
    return extracted


def preprocess_original_img(
    image_path: str | Path,
    target_shape: tuple[int, int, int] = (128, 128, 96),
    clip_percentiles: tuple[float, float] = (0.5, 99.5),
    crop_margin: int = 12,
    final_clip: tuple[float, float] = (-5.0, 5.0),
    final_scale: float = 5.0,
):
    """Versioned copy of the accepted no-external-mask preprocessing route."""

    import nibabel as nib
    import numpy as np
    from scipy.ndimage import zoom

    image = nib.load(str(image_path))
    volume = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    mask = volume != 0
    coordinates = np.argwhere(mask)
    if coordinates.size:
        lower = np.maximum(coordinates.min(axis=0) - crop_margin, 0)
        upper = np.minimum(coordinates.max(axis=0) + crop_margin + 1, mask.shape)
        slices = tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))
        volume = volume[slices]
        mask = mask[slices]
    foreground = volume[mask]
    if foreground.size == 0:
        foreground = volume.reshape(-1)
    if foreground.size == 0:
        return np.zeros(target_shape, dtype=np.float32)
    low, high = np.percentile(foreground, clip_percentiles)
    if np.isfinite(low) and np.isfinite(high) and high > low:
        volume = np.clip(volume, low, high)
    foreground = volume[mask] if mask.any() else volume.reshape(-1)
    mean = float(foreground.mean())
    standard_deviation = float(foreground.std())
    if not np.isfinite(standard_deviation) or standard_deviation < 1e-6:
        standard_deviation = 1.0
    volume = (volume - mean) / standard_deviation
    volume = np.clip(volume, final_clip[0], final_clip[1])
    volume = volume / float(final_scale)
    factors = [target / source for target, source in zip(target_shape, volume.shape)]
    volume = zoom(volume, factors, order=1).astype(np.float32, copy=False)
    return volume.astype(np.float32, copy=False)


def _tuple_to_json(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_tuple_to_json(item) for item in value]
    if isinstance(value, list):
        return [_tuple_to_json(item) for item in value]
    return value


def _json_to_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_json_to_tuple(item) for item in value)
    return value


def capture_rng_state(*, include_torch: bool = True) -> dict[str, Any]:
    """Capture Python, NumPy, and optionally Torch RNG without pickle."""

    import numpy as np

    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": _tuple_to_json(random.getstate()),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": numpy_state[1].astype("uint32", copy=False).tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
    }
    if include_torch:
        import torch

        state["torch_cpu_base64"] = base64.b64encode(
            bytes(torch.get_rng_state().cpu().tolist())
        ).decode("ascii")
        cuda_states: list[str] = []
        if torch.cuda.is_available():
            cuda_states = [
                base64.b64encode(bytes(item.cpu().tolist())).decode("ascii")
                for item in torch.cuda.get_rng_state_all()
            ]
        state["torch_cuda_base64"] = cuda_states
    return state


def restore_rng_state(state: Mapping[str, Any], *, include_torch: bool = True) -> None:
    import numpy as np

    random.setstate(_json_to_tuple(state["python"]))
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["algorithm"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    if include_torch:
        import torch

        cpu_bytes = base64.b64decode(state["torch_cpu_base64"], validate=True)
        torch.set_rng_state(torch.tensor(list(cpu_bytes), dtype=torch.uint8))
        cuda_payloads = list(state.get("torch_cuda_base64", []))
        if cuda_payloads:
            if not torch.cuda.is_available() or len(cuda_payloads) != torch.cuda.device_count():
                raise RuntimeContractError("Checkpoint Torch CUDA RNG topology changed")
            torch.cuda.set_rng_state_all(
                [
                    torch.tensor(
                        list(base64.b64decode(payload, validate=True)), dtype=torch.uint8
                    )
                    for payload in cuda_payloads
                ]
            )


def validate_checkpoint_state(
    state: Mapping[str, Any],
    *,
    expected_candidate: str | None = None,
    expected_input_fingerprint: str | None = None,
) -> None:
    required = {
        "schema_version",
        "candidate_id",
        "input_fingerprint",
        "case_uids",
        "stage",
        "next_case_index",
        "probabilities",
        "labels",
        "reports",
        "rng_state",
    }
    if set(state) != required:
        raise RuntimeContractError(
            f"Checkpoint keys differ: missing={sorted(required - set(state))}, "
            f"extra={sorted(set(state) - required)}"
        )
    if state["schema_version"] != SCHEMA_VERSION:
        raise RuntimeContractError("Checkpoint schema version changed")
    if expected_candidate is not None and state["candidate_id"] != expected_candidate:
        raise RuntimeContractError("Checkpoint candidate changed")
    fingerprint = str(state["input_fingerprint"])
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise RuntimeContractError("Checkpoint input fingerprint is malformed")
    if expected_input_fingerprint is not None and fingerprint != expected_input_fingerprint:
        raise RuntimeContractError("Checkpoint input inventory changed")
    uids = list(state["case_uids"])
    if not uids or len(uids) != len(set(uids)) or any(
        not isinstance(uid, str) or not UID_PATTERN.fullmatch(uid) for uid in uids
    ):
        raise RuntimeContractError("Checkpoint UID inventory is malformed")
    probabilities = list(state["probabilities"])
    labels = list(state["labels"])
    reports = list(state["reports"])
    stage = state["stage"]
    next_index = state["next_case_index"]
    if not isinstance(next_index, int) or not 0 <= next_index <= len(uids):
        raise RuntimeContractError("Checkpoint next-case index is malformed")
    if len(probabilities) != len(labels):
        raise RuntimeContractError("Checkpoint probability/label lengths differ")
    if stage == "vision":
        if len(probabilities) != next_index or reports:
            raise RuntimeContractError("Vision-stage checkpoint lengths are inconsistent")
    elif stage == "report":
        if len(probabilities) != len(uids) or len(reports) != next_index:
            raise RuntimeContractError("Report-stage checkpoint lengths are inconsistent")
    elif stage == "complete":
        if (
            next_index != len(uids)
            or len(probabilities) != len(uids)
            or len(reports) != len(uids)
        ):
            raise RuntimeContractError("Complete checkpoint lengths are inconsistent")
    else:
        raise RuntimeContractError(f"Unsupported checkpoint stage: {stage!r}")
    for index, report in enumerate(reports):
        expected_uid = uids[index]
        if set(report) != {"uid", "input_image_name", "report"}:
            raise RuntimeContractError("Checkpoint report record keys changed")
        if report["uid"] != expected_uid or report["input_image_name"] != f"{expected_uid}.nii.gz":
            raise RuntimeContractError("Checkpoint report order or output name changed")
        if not isinstance(report["report"], str) or not report["report"].strip():
            raise RuntimeContractError("Checkpoint contains an empty report")
    if not isinstance(state["rng_state"], dict):
        raise RuntimeContractError("Checkpoint RNG state is malformed")


class AtomicCheckpointStore:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        expected_candidate: str = CANDIDATE_ID,
        expected_input_fingerprint: str,
    ) -> None:
        self.path = Path(checkpoint_path)
        self.expected_candidate = expected_candidate
        self.expected_input_fingerprint = expected_input_fingerprint

    def save(self, state: Mapping[str, Any]) -> None:
        validate_checkpoint_state(
            state,
            expected_candidate=self.expected_candidate,
            expected_input_fingerprint=self.expected_input_fingerprint,
        )
        atomic_write_json(self.path, state)

    def load(self) -> dict[str, Any]:
        _require_regular_no_symlink(self.path, "Checkpoint")
        state = json.loads(self.path.read_text(encoding="utf-8"))
        validate_checkpoint_state(
            state,
            expected_candidate=self.expected_candidate,
            expected_input_fingerprint=self.expected_input_fingerprint,
        )
        return state


def build_predictions(state: Mapping[str, Any]) -> dict[str, Any]:
    validate_checkpoint_state(state, expected_candidate=str(state["candidate_id"]))
    if state["stage"] != "complete":
        raise RuntimeContractError("Predictions may only be emitted from a complete checkpoint")
    generated = [
        {"input_image_name": record["input_image_name"], "report": record["report"]}
        for record in state["reports"]
    ]
    return {"generated_reports": generated}


def write_predictions(path: str | Path, state: Mapping[str, Any]) -> None:
    atomic_write_json(path, build_predictions(state))


def install_sigterm_checkpoint(checkpoint_callback: Callable[[], None]):
    """Save the last committed state synchronously and exit on SIGTERM."""

    saving = False

    def handler(_signum, _frame):
        nonlocal saving
        if saving:
            raise SystemExit(0)
        saving = True
        checkpoint_callback()
        raise SystemExit(0)

    return signal.signal(signal.SIGTERM, handler)
