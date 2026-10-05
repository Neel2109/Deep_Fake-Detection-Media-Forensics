"""Tests for durable SQLite case and report records."""

from pathlib import Path
import tempfile
import unittest

from database.connection import (
    initialize_database,
    load_cases,
    load_reports,
    save_analysis,
    save_report,
)


class DatabasePersistenceTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.database_path = Path(temporary_directory.name) / "deeptrace.sqlite3"
        initialize_database(self.database_path)

    def test_case_and_report_survive_database_reinitialization(self):
        case = {
            "case_id": "case-persisted",
            "title": "Evidence review",
            "created_at": "2026-10-04T12:00:00+00:00",
        }
        report = {
            "case_id": "case-persisted",
            "filename": "evidence.png",
            "created_at": "2026-10-04T12:00:00+00:00",
            "frequency_analysis": {"status": "measured_descriptive"},
        }

        save_analysis(report, case, self.database_path)
        initialize_database(self.database_path)

        self.assertEqual(load_cases(self.database_path), [case])
        self.assertEqual(load_reports(self.database_path), [report])

    def test_report_update_preserves_nested_analyst_review(self):
        case = {
            "case_id": "case-reviewed",
            "title": "Review",
            "created_at": "2026-10-04T12:00:00+00:00",
        }
        report = {
            "case_id": "case-reviewed",
            "filename": "evidence.png",
            "created_at": "2026-10-04T12:00:00+00:00",
        }
        save_analysis(report, case, self.database_path)
        updated_report = {
            **report,
            "analyst_review": {
                "outcome": "inconclusive",
                "rationale": "Insufficient source material",
            },
        }

        save_report(updated_report, self.database_path)

        self.assertEqual(load_reports(self.database_path), [updated_report])

    def test_analysis_case_and_report_are_saved_atomically(self):
        invalid_report = {
            "case_id": "case-invalid",
            "created_at": "2026-10-04T12:00:00+00:00",
            "non_finite": float("nan"),
        }
        case = {
            "case_id": "case-invalid",
            "created_at": "2026-10-04T12:00:00+00:00",
        }

        with self.assertRaises(ValueError):
            save_analysis(invalid_report, case, self.database_path)

        self.assertEqual(load_cases(self.database_path), [])
        self.assertEqual(load_reports(self.database_path), [])


if __name__ == "__main__":
    unittest.main()
