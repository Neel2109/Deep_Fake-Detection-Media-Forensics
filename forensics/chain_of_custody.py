"""Tamper-evident hash chaining for report audit events."""

import hashlib
import json
from typing import Any

GENESIS_EVENT_SHA256 = "0" * 64
CHAIN_ALGORITHM = "sha256-json-v1"


def _event_digest(
    case_id: str,
    evidence_sha256: str,
    event: dict[str, Any],
) -> str:
    payload = {
        "case_id": case_id,
        "evidence_sha256": evidence_sha256,
        "event": {
            key: value
            for key, value in event.items()
            if key != "event_sha256"
        },
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _seal_event(
    case_id: str,
    evidence_sha256: str,
    event: dict[str, Any],
    previous_hash: str,
) -> dict[str, Any]:
    sealed = {
        key: value
        for key, value in event.items()
        if key not in {"previous_event_sha256", "event_sha256"}
    }
    sealed.setdefault("event_type", "forensic_event")
    sealed["previous_event_sha256"] = previous_hash
    sealed["event_sha256"] = _event_digest(case_id, evidence_sha256, sealed)
    return sealed


def _update_chain_metadata(report: dict[str, Any]) -> None:
    events = report.get("audit_log", [])
    report["chain_of_custody"] = {
        "algorithm": CHAIN_ALGORITHM,
        "event_count": len(events),
        "head_sha256": events[-1]["event_sha256"] if events else GENESIS_EVENT_SHA256,
        "status": "tamper_evident",
        "storage_limitation": (
            "Events are hash-linked inside mutable SQLite report storage; this is not an immutable ledger."
        ),
    }


def seal_audit_log(report: dict[str, Any]) -> None:
    """Seal initial report audit entries, binding each event to the case and evidence hash."""
    case_id = str(report["case_id"])
    evidence_sha256 = str(report["file_hash"])
    previous_hash = GENESIS_EVENT_SHA256
    sealed_events = []
    for event in report.get("audit_log", []):
        sealed = _seal_event(case_id, evidence_sha256, event, previous_hash)
        sealed_events.append(sealed)
        previous_hash = sealed["event_sha256"]
    report["audit_log"] = sealed_events
    _update_chain_metadata(report)


def verify_audit_chain(report: dict[str, Any]) -> bool:
    """Verify event order, event digests, evidence binding, and the recorded chain head."""
    chain = report.get("chain_of_custody")
    events = report.get("audit_log")
    if not isinstance(chain, dict) or not isinstance(events, list):
        return False
    if chain.get("algorithm") != CHAIN_ALGORITHM or chain.get("event_count") != len(events):
        return False
    previous_hash = GENESIS_EVENT_SHA256
    try:
        for event in events:
            if not isinstance(event, dict):
                return False
            if event.get("previous_event_sha256") != previous_hash:
                return False
            actual_hash = event.get("event_sha256")
            if not isinstance(actual_hash, str) or actual_hash != _event_digest(
                str(report["case_id"]),
                str(report["file_hash"]),
                event,
            ):
                return False
            previous_hash = actual_hash
    except (KeyError, TypeError, ValueError):
        return False
    expected_head = events[-1]["event_sha256"] if events else GENESIS_EVENT_SHA256
    return chain.get("head_sha256") == expected_head


def append_audit_event(
    report: dict[str, Any],
    *,
    event_type: str,
    timestamp: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Append a chained event, upgrading unchained legacy logs but rejecting broken chains."""
    events = report.get("audit_log")
    if not isinstance(events, list):
        raise ValueError("Report audit_log must be a list")

    if not report.get("chain_of_custody"):
        if any("event_sha256" in event for event in events if isinstance(event, dict)):
            raise ValueError("Cannot append to a partially chained audit log")
        seal_audit_log(report)
    elif not verify_audit_chain(report):
        raise ValueError("Cannot append to an invalid audit chain")

    previous_hash = (
        report["audit_log"][-1]["event_sha256"]
        if report["audit_log"]
        else GENESIS_EVENT_SHA256
    )
    event: dict[str, Any] = {
        "event_type": event_type,
        "timestamp": timestamp,
        "message": message,
    }
    if details:
        event["details"] = details
    sealed = _seal_event(
        str(report["case_id"]),
        str(report["file_hash"]),
        event,
        previous_hash,
    )
    report["audit_log"].append(sealed)
    _update_chain_metadata(report)
