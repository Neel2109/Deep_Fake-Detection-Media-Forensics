"""Small SQLite repository for cases and forensic analysis reports."""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator

from app.config import DATABASE_PATH


def _connect(path: str | Path | None = None) -> sqlite3.Connection:
    database_path = Path(path) if path is not None else DATABASE_PATH
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    return connection


@contextmanager
def _transaction(path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    connection = _connect(path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database(path: str | Path | None = None) -> None:
    with _transaction(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                case_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                FOREIGN KEY (case_id) REFERENCES cases(case_id)
                    ON UPDATE CASCADE ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports(created_at)"
        )


def _serialize_record(record: dict[str, Any]) -> tuple[str, str, str]:
    record_id = record.get("case_id")
    created_at = record.get("created_at")
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("Database records require a non-empty case_id")
    if not isinstance(created_at, str) or not created_at.strip():
        raise ValueError("Database records require a non-empty created_at timestamp")
    payload = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return record_id, created_at, payload


def _save_case(connection: sqlite3.Connection, case: dict[str, Any]) -> None:
    case_id, created_at, payload = _serialize_record(case)
    connection.execute(
        """
        INSERT INTO cases (case_id, created_at, payload_json)
        VALUES (?, ?, ?)
        ON CONFLICT(case_id) DO UPDATE SET
            created_at = excluded.created_at,
            payload_json = excluded.payload_json
        """,
        (case_id, created_at, payload),
    )


def _save_report(connection: sqlite3.Connection, report: dict[str, Any]) -> None:
    case_id, created_at, payload = _serialize_record(report)
    connection.execute(
        """
        INSERT INTO reports (case_id, created_at, payload_json)
        VALUES (?, ?, ?)
        ON CONFLICT(case_id) DO UPDATE SET
            created_at = excluded.created_at,
            payload_json = excluded.payload_json
        """,
        (case_id, created_at, payload),
    )


def save_case(case: dict[str, Any], path: str | Path | None = None) -> None:
    with _transaction(path) as connection:
        _save_case(connection, case)


def save_analysis(
    report: dict[str, Any],
    case: dict[str, Any],
    path: str | Path | None = None,
) -> None:
    with _transaction(path) as connection:
        _save_case(connection, case)
        _save_report(connection, report)


def save_report(report: dict[str, Any], path: str | Path | None = None) -> None:
    with _transaction(path) as connection:
        _save_report(connection, report)


def _load_records(
    table: str,
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    if table not in {"cases", "reports"}:
        raise ValueError(f"Unsupported record table: {table}")
    with _transaction(path) as connection:
        rows = connection.execute(
            f"SELECT payload_json FROM {table} ORDER BY rowid"
        ).fetchall()
    records = [json.loads(row["payload_json"]) for row in rows]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"Stored {table} payload must be a JSON object")
    return records


def load_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    return _load_records("cases", path)


def load_reports(path: str | Path | None = None) -> list[dict[str, Any]]:
    return _load_records("reports", path)
