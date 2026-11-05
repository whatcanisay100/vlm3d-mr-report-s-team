from __future__ import annotations
import json
from typing import Any



def append_jsonl(handle: Any, rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        handle.write(canonical_json_bytes(row))
    handle.flush()
    os.fsync(handle.fileno())


def report_record(
    base: Any,
    evaluator: Any,
    templates: dict[str, str],
    uid: str,
    labels: list[str],
    decoded: str,
    token_count: int,
    cap_hit: bool,
) -> dict[str, Any]:
    report = base.clean_output(decoded)
    lower = report.casefold()
    format_valid = bool(
        report and "technique" in lower and "findings" in lower and "impression" in lower
    )
    extracted_before = evaluator.extract_labels_37(report)
    fallback_labels = [label for label in labels if not extracted_before.get(label, 0)]
    if fallback_labels:
        report = base.append_coverage_fallback(report, fallback_labels, templates)
    extracted_after = evaluator.extract_labels_37(report)
    missing = [label for label in labels if not extracted_after.get(label, 0)]
    return {
        "uid": uid,
        "report": report,
        "input_report_labels": labels,
        "fallback_labels": fallback_labels,
        "generated_tokens": token_count,
        "max_new_tokens_hit": cap_hit,
        "format_valid": format_valid,
        "missing_positive_coverage": missing,
        "valid": bool(format_valid and not cap_hit and not missing),
    }


def decode_rows(
    base: Any,
    torch: Any,
    tokenizer: Any,
    model: Any,
    evaluator: Any,
    templates: dict[str, str],
    rows: list[dict[str, Any]],
    arm: str,
    no_repeat_ngram_size: int | None,
    jsonl_handle: Any,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for start in range(0, len(rows), base.LLM_BATCH):
        batch = rows[start : start + base.LLM_BATCH]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": base.user_prompt(row["report_labels"])}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)
        input_length = encoded["input_ids"].shape[1]
        kwargs: dict[str, Any] = {
            "max_new_tokens": base.MAX_NEW_TOKENS,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if no_repeat_ngram_size is not None:
            kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        with torch.inference_mode():
            generated = model.generate(**encoded, **kwargs)
        batch_records: list[dict[str, Any]] = []
        for row, tokens in zip(batch, generated, strict=True):
            continuation = tokens[input_length:]
            eos_positions = torch.nonzero(
                continuation == tokenizer.eos_token_id, as_tuple=False
            ).flatten()
            token_count = (
                int(eos_positions[0].item()) + 1
                if len(eos_positions)
                else int(len(continuation))
            )
            cap_hit = not len(eos_positions) and token_count >= base.MAX_NEW_TOKENS
            record = report_record(
                base,
                evaluator,
                templates,
                row["uid"],
                row["report_labels"],
                tokenizer.decode(continuation[:token_count], skip_special_tokens=True),
                token_count,
                cap_hit,
            )
            record["arm"] = arm
            record["group"] = row.get("group", "public_generation")
            record["no_repeat_ngram_size"] = no_repeat_ngram_size
            batch_records.append(record)
        append_jsonl(jsonl_handle, batch_records)
        output.extend(batch_records)
        print(
            json.dumps(
                {
                    "arm": arm,
                    "decoded": len(output),
                    "total": len(rows),
                    "valid": sum(record["valid"] for record in output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return output
