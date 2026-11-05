
"""Versioned every-volume release I/O and checkpoint boundary for w045.

This module intentionally does not load the visual encoders or report model.
The model backend must provide one explicit record for every case routed to the
frozen candidate.  Data-specific failures may be marked unusable; absent model
records and model failures remain fatal.  That separation prevents a model
failure from being silently converted into a technical-limitation report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

try:
    from docker.forithmus_v2 import runtime_io
except ModuleNotFoundError:
    
    
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import runtime_io  


SCHEMA_VERSION = "w045_release_checkpoint_v1"
ROUTE_SCHEMA_VERSION = "w045_every_volume_route_v1"
CANDIDATE_RECORD_SCHEMA_VERSION = "w045_candidate_records_v1"
CANDIDATE_ID = (
    "w045_proxy32_functional_confirm_v1__qwen2.5_14b_fp16__lora_v2__"
    "greedy_no_repeat_ngram6__positive_coverage_v1"
)
TECHNICAL_LIMITATION_REPORT = (
    "Technique: Submitted MRI data were reviewed for technical adequacy.\n"
    "Findings: The available images are incomplete or technically unusable, "
    "so a diagnostic interpretation cannot be provided.\n"
    "Impression: Nondiagnostic examination due to incomplete or technically "
    "unusable MRI data. Repeat imaging and clinical correlation are recommended."
)
DATA_FAILURE_REASONS = {
    "archive_unusable",
    "missing_required_modality",
    "preprocess_unusable",
}
CHECKPOINT_KEYS = {
    "schema_version",
    "candidate_id",
    "input_fingerprint",
    "case_uids",
    "next_case_index",
    "records",
}
RECORD_KEYS = {
    "uid",
    "input_image_name",
    "route",
    "reason_code",
    "probabilities",
    "labels",
    "report",
}


class ReleaseRuntimeError(RuntimeError):
    """Raised when the every-volume release contract is violated."""


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ReleaseRuntimeError(f"JSON input must be a regular non-symlink file: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReleaseRuntimeError(f"JSON input must contain an object: {source}")
    return payload


def _report_is_well_formed(report: Any) -> bool:
    if not isinstance(report, str) or not report.strip():
        return False
    lowered = report.casefold()
    return all(section in lowered for section in ("technique", "findings", "impression"))


def _probabilities_and_labels_are_valid(probabilities: Any, labels: Any) -> bool:
    return bool(
        isinstance(probabilities, list)
        and len(probabilities) == 32
        and not any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in probabilities
        )
        and isinstance(labels, list)
        and len(labels) == len(set(labels))
        and all(isinstance(label, str) and bool(label) for label in labels)
    )


def build_route_plan(input_root: str | Path) -> dict[str, Any]:
    """Resolve all declared inputs and deterministically select their route."""

    cases = runtime_io.discover_cases(input_root)
    fingerprint = runtime_io.input_inventory_fingerprint(cases)
    records: list[dict[str, Any]] = []
    for case in cases:
        try:
            selection = runtime_io.scan_case_zip(case)
        except (
            runtime_io.RuntimeContractError,
            OSError,
            ValueError,
            zipfile.BadZipFile,
            zipfile.LargeZipFile,
        ) as error:
            records.append(
                {
                    "uid": case.uid,
                    "archive_name": case.archive.name,
                    "input_image_name": case.input_image_name,
                    "route": "technical_limitation",
                    "reason_code": "archive_unusable",
                    "selected": {},
                    "candidate_counts": {},
                    "diagnostic_error_type": type(error).__name__,
                }
            )
            continue
        selected = {
            modality: PurePosixPath(member).name
            for modality, member in selection.selected.items()
        }
        missing = sorted(set(runtime_io.REQUIRED_MODALITIES) - set(selected))
        route = "candidate" if not missing else "technical_limitation"
        records.append(
            {
                "uid": case.uid,
                "archive_name": case.archive.name,
                "input_image_name": case.input_image_name,
                "route": route,
                "reason_code": None if route == "candidate" else "missing_required_modality",
                "selected": selected,
                "candidate_counts": {
                    modality: len(members)
                    for modality, members in selection.candidates.items()
                },
                "diagnostic_error_type": None,
            }
        )
    plan = {
        "schema_version": ROUTE_SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "input_fingerprint": fingerprint,
        "case_uids": [case.uid for case in cases],
        "records": records,
    }
    validate_route_plan(plan)
    return plan


def validate_route_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != ROUTE_SCHEMA_VERSION:
        raise ReleaseRuntimeError("Route-plan schema changed")
    if plan.get("candidate_id") != CANDIDATE_ID:
        raise ReleaseRuntimeError("Route-plan candidate changed")
    fingerprint = plan.get("input_fingerprint")
    if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise ReleaseRuntimeError("Route-plan input fingerprint is malformed")
    uids = plan.get("case_uids")
    records = plan.get("records")
    if (
        not isinstance(uids, list)
        or not uids
        or len(uids) != len(set(uids))
        or not isinstance(records, list)
        or len(records) != len(uids)
        or [record.get("uid") for record in records] != uids
    ):
        raise ReleaseRuntimeError("Route-plan UID/order contract changed")
    for record in records:
        if record.get("route") not in {"candidate", "technical_limitation"}:
            raise ReleaseRuntimeError("Unsupported every-volume route")
        reason = record.get("reason_code")
        if record["route"] == "candidate" and reason is not None:
            raise ReleaseRuntimeError("Candidate route unexpectedly has a failure reason")
        if record["route"] == "technical_limitation" and reason not in DATA_FAILURE_REASONS:
            raise ReleaseRuntimeError("Technical-limitation route has an unregistered reason")


def validate_candidate_payload(
    payload: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    validate_route_plan(plan)
    if payload.get("schema_version") != CANDIDATE_RECORD_SCHEMA_VERSION:
        raise ReleaseRuntimeError("Candidate-record schema changed")
    if payload.get("candidate_id") != CANDIDATE_ID:
        raise ReleaseRuntimeError("Candidate-record model identity changed")
    if payload.get("input_fingerprint") != plan["input_fingerprint"]:
        raise ReleaseRuntimeError("Candidate records target a different input inventory")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ReleaseRuntimeError("Candidate records must be a list")
    by_uid: dict[str, dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get("uid"), str):
            raise ReleaseRuntimeError("Malformed candidate record")
        uid = item["uid"]
        if uid in by_uid:
            raise ReleaseRuntimeError(f"Duplicate candidate record: {uid}")
        status = item.get("status")
        if status == "complete":
            probabilities = item.get("probabilities")
            labels = item.get("labels")
            if (
                not _probabilities_and_labels_are_valid(probabilities, labels)
                or not _report_is_well_formed(item.get("report"))
                or item.get("reason_code") is not None
            ):
                raise ReleaseRuntimeError(f"Malformed completed candidate record: {uid}")
        elif status == "unusable":
            if (
                item.get("reason_code") != "preprocess_unusable"
                or item.get("probabilities") is not None
                or item.get("labels") != []
                or item.get("report") is not None
            ):
                raise ReleaseRuntimeError(f"Malformed unusable candidate record: {uid}")
        else:
            raise ReleaseRuntimeError(f"Unsupported candidate-record status: {uid}/{status}")
        by_uid[uid] = item
    expected = [
        record["uid"] for record in plan["records"] if record["route"] == "candidate"
    ]
    if list(by_uid) != expected:
        raise ReleaseRuntimeError(
            f"Candidate-record UID/order differs: observed={list(by_uid)} expected={expected}"
        )
    return by_uid


def initial_checkpoint(plan: Mapping[str, Any]) -> dict[str, Any]:
    validate_route_plan(plan)
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "input_fingerprint": plan["input_fingerprint"],
        "case_uids": list(plan["case_uids"]),
        "next_case_index": 0,
        "records": [],
    }


def validate_checkpoint(state: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    validate_route_plan(plan)
    if set(state) != CHECKPOINT_KEYS:
        raise ReleaseRuntimeError("Release checkpoint keys changed")
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("candidate_id") != CANDIDATE_ID
        or state.get("input_fingerprint") != plan["input_fingerprint"]
        or state.get("case_uids") != plan["case_uids"]
    ):
        raise ReleaseRuntimeError("Release checkpoint provenance changed")
    next_index = state.get("next_case_index")
    records = state.get("records")
    if (
        not isinstance(next_index, int)
        or not 0 <= next_index <= len(plan["case_uids"])
        or not isinstance(records, list)
        or len(records) != next_index
    ):
        raise ReleaseRuntimeError("Release checkpoint progress is malformed")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != RECORD_KEYS:
            raise ReleaseRuntimeError("Release checkpoint record schema changed")
        uid = plan["case_uids"][index]
        planned = plan["records"][index]
        if (
            record["uid"] != uid
            or record["input_image_name"] != f"{uid}.nii.gz"
            or record["route"] not in {"candidate", "technical_limitation"}
            or not _report_is_well_formed(record["report"])
        ):
            raise ReleaseRuntimeError("Release checkpoint record order/content changed")
        if record["route"] == "technical_limitation":
            if (
                record["reason_code"] not in DATA_FAILURE_REASONS
                or record["probabilities"] is not None
                or record["labels"] != []
                or record["report"] != TECHNICAL_LIMITATION_REPORT
            ):
                raise ReleaseRuntimeError("Technical-limitation checkpoint record changed")
            if planned["route"] == "technical_limitation":
                if record["reason_code"] != planned["reason_code"]:
                    raise ReleaseRuntimeError("Route-level technical-limitation reason changed")
            elif record["reason_code"] != "preprocess_unusable":
                raise ReleaseRuntimeError("Candidate route used an unauthorized fallback reason")
        else:
            if (
                planned["route"] != "candidate"
                or record["reason_code"] is not None
                or not _probabilities_and_labels_are_valid(
                    record["probabilities"], record["labels"]
                )
            ):
                raise ReleaseRuntimeError("Candidate checkpoint record changed")


class ReleaseCheckpointStore:
    def __init__(self, path: str | Path, plan: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.plan = dict(plan)

    def save(self, state: Mapping[str, Any]) -> None:
        validate_checkpoint(state, self.plan)
        runtime_io.atomic_write_json(self.path, state)

    def load(self) -> dict[str, Any]:
        state = _load_json(self.path)
        validate_checkpoint(state, self.plan)
        return state


def _completed_record(uid: str, item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "uid": uid,
        "input_image_name": f"{uid}.nii.gz",
        "route": "candidate",
        "reason_code": None,
        "probabilities": [float(value) for value in item["probabilities"]],
        "labels": list(item["labels"]),
        "report": item["report"],
    }


def _fallback_record(uid: str, reason_code: str) -> dict[str, Any]:
    if reason_code not in DATA_FAILURE_REASONS:
        raise ReleaseRuntimeError(f"Unregistered technical-limitation reason: {reason_code}")
    return {
        "uid": uid,
        "input_image_name": f"{uid}.nii.gz",
        "route": "technical_limitation",
        "reason_code": reason_code,
        "probabilities": None,
        "labels": [],
        "report": TECHNICAL_LIMITATION_REPORT,
    }


def finalize_every_volume(
    plan: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    stop_after: int | None = None,
) -> dict[str, Any]:
    """Resume or finish every-volume assembly from model-backend records."""

    validate_route_plan(plan)
    candidates = validate_candidate_payload(candidate_payload, plan)
    store = ReleaseCheckpointStore(checkpoint_path, plan)
    if store.path.exists() or store.path.is_symlink():
        state = store.load()
    else:
        state = initial_checkpoint(plan)
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing release output: {output}")
    for index in range(state["next_case_index"], len(plan["records"])):
        route_record = plan["records"][index]
        uid = route_record["uid"]
        if route_record["route"] == "technical_limitation":
            record = _fallback_record(uid, route_record["reason_code"])
        else:
            model_record = candidates[uid]
            if model_record["status"] == "unusable":
                record = _fallback_record(uid, model_record["reason_code"])
            else:
                record = _completed_record(uid, model_record)
        state["records"].append(record)
        state["next_case_index"] = index + 1
        store.save(state)
        if stop_after is not None and state["next_case_index"] >= stop_after:
            return {
                "status": "interrupted_for_smoke",
                "next_case_index": state["next_case_index"],
                "total_cases": len(plan["case_uids"]),
                "output_written": False,
            }
    validate_checkpoint(state, plan)
    predictions = {
        "generated_reports": [
            {
                "input_image_name": record["input_image_name"],
                "report": record["report"],
            }
            for record in state["records"]
        ]
    }
    runtime_io.atomic_write_json(output, predictions)
    return {
        "status": "complete",
        "next_case_index": state["next_case_index"],
        "total_cases": len(plan["case_uids"]),
        "candidate_reports": sum(record["route"] == "candidate" for record in state["records"]),
        "technical_limitation_reports": sum(
            record["route"] == "technical_limitation" for record in state["records"]
        ),
        "reports_nonempty": sum(bool(record["report"].strip()) for record in state["records"]),
        "output_written": True,
    }


def _write_route(path: Path, plan: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing existing route-plan output: {path}")
    runtime_io.atomic_write_json(path, plan)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("route", "finalize", "sigterm-probe"))
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--route-json", type=Path)
    parser.add_argument("--candidate-records", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--ready-marker", type=Path)
    args = parser.parse_args()
    plan = build_route_plan(args.input_root)
    if args.mode == "route":
        if args.route_json is None or any(
            value is not None
            for value in (args.candidate_records, args.checkpoint, args.output, args.stop_after, args.ready_marker)
        ):
            raise ReleaseRuntimeError("route mode accepts only --input-root/--route-json")
        _write_route(args.route_json, plan)
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
        return
    if args.route_json is not None:
        persisted_plan = _load_json(args.route_json)
        if persisted_plan != plan:
            raise ReleaseRuntimeError("Persisted route plan differs from live input")
    if args.mode == "finalize":
        if args.candidate_records is None or args.checkpoint is None or args.output is None:
            raise ReleaseRuntimeError("finalize mode requires candidate records, checkpoint, and output")
        if args.ready_marker is not None:
            raise ReleaseRuntimeError("finalize mode does not accept --ready-marker")
        result = finalize_every_volume(
            plan,
            _load_json(args.candidate_records),
            args.checkpoint,
            args.output,
            stop_after=args.stop_after,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return
    if args.checkpoint is None or args.ready_marker is None:
        raise ReleaseRuntimeError("sigterm-probe mode requires checkpoint and ready marker")
    if any(value is not None for value in (args.candidate_records, args.output, args.stop_after)):
        raise ReleaseRuntimeError("sigterm-probe mode accepts no candidate/output/stop controls")
    store = ReleaseCheckpointStore(args.checkpoint, plan)
    state = store.load()
    runtime_io.install_sigterm_checkpoint(lambda: store.save(state))
    runtime_io.atomic_write_json(args.ready_marker, {"ready": True, "pid": os.getpid()})
    while True:
        signal.pause()


if __name__ == "__main__":
    main()
