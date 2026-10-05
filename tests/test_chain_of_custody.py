import copy

import pytest

from forensics.chain_of_custody import append_audit_event, seal_audit_log, verify_audit_chain


def make_report():
    return {
        "case_id": "case-1",
        "file_hash": "a" * 64,
        "created_at": "2026-10-05T00:00:00+00:00",
        "audit_log": [
            {
                "event_type": "upload_received",
                "timestamp": "2026-10-05T00:00:00+00:00",
                "message": "Evidence received.",
            },
        ],
    }


def test_sealed_audit_log_binds_events_to_evidence_and_verifies():
    report = make_report()
    seal_audit_log(report)

    assert verify_audit_chain(report)
    assert report["chain_of_custody"]["event_count"] == 1
    assert report["audit_log"][0]["previous_event_sha256"] == "0" * 64


def test_audit_chain_detects_event_edits_reordering_and_evidence_changes():
    report = make_report()
    seal_audit_log(report)

    edited_event = copy.deepcopy(report)
    edited_event["audit_log"][0]["message"] = "Changed."
    assert not verify_audit_chain(edited_event)

    changed_evidence = copy.deepcopy(report)
    changed_evidence["file_hash"] = "b" * 64
    assert not verify_audit_chain(changed_evidence)


def test_appending_event_extends_chain_and_detects_legacy_logs():
    report = make_report()
    append_audit_event(
        report,
        event_type="analyst_review_recorded",
        timestamp="2026-10-05T00:01:00+00:00",
        message="Review recorded.",
        details={"outcome": "inconclusive"},
    )

    assert verify_audit_chain(report)
    assert report["chain_of_custody"]["event_count"] == 2
    assert (
        report["audit_log"][1]["previous_event_sha256"]
        == report["audit_log"][0]["event_sha256"]
    )


def test_appending_refuses_a_previously_tampered_chain():
    report = make_report()
    seal_audit_log(report)
    report["audit_log"][0]["message"] = "Tampered."

    with pytest.raises(ValueError, match="invalid audit chain"):
        append_audit_event(
            report,
            event_type="analyst_review_recorded",
            timestamp="2026-10-05T00:01:00+00:00",
            message="Review recorded.",
        )
