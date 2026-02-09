"""Tests for telemetry report generation."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Import functions to test - using absolute imports
import sys
from pathlib import Path as PathLib

# Add scripts directory to path to import telemetry_report
scripts_dir = PathLib(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from telemetry_report import (
    load_events,
    is_report_empty,
    generate_text_report,
    generate_html_report,
    send_email,
)


class TestLoadEvents:
    """Tests for load_events function."""

    def test_load_events_no_directory(self, tmp_path):
        """Test loading events when directory doesn't exist."""
        telemetry_dir = tmp_path / "nonexistent"
        events = load_events(telemetry_dir, "session")
        assert events == []

    def test_load_events_empty_directory(self, tmp_path):
        """Test loading events from empty directory."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        events = load_events(tmp_path, "session")
        assert events == []

    def test_load_events_all_files(self, tmp_path):
        """Test loading all events without date filter."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        # Create test files
        events_day1 = [
            {"event_type": "session", "timestamp": "2026-01-20T10:00:00Z"},
            {"event_type": "session", "timestamp": "2026-01-20T11:00:00Z"},
        ]
        events_day2 = [
            {"event_type": "session", "timestamp": "2026-01-21T10:00:00Z"},
        ]

        file1 = daily_dir / "session_2026-01-20.jsonl"
        file2 = daily_dir / "session_2026-01-21.jsonl"

        with open(file1, "w", encoding="utf-8") as f:
            for event in events_day1:
                f.write(json.dumps(event) + "\n")

        with open(file2, "w", encoding="utf-8") as f:
            for event in events_day2:
                f.write(json.dumps(event) + "\n")

        # Load all events
        events = load_events(tmp_path, "session")
        assert len(events) == 3

    def test_load_events_specific_date(self, tmp_path):
        """Test loading events for specific date."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        # Create test files
        file1 = daily_dir / "usage_2026-01-20.jsonl"
        file2 = daily_dir / "usage_2026-01-21.jsonl"

        with open(file1, "w", encoding="utf-8") as f:
            f.write(json.dumps({"event_type": "usage", "slide_count": 1}) + "\n")

        with open(file2, "w", encoding="utf-8") as f:
            f.write(json.dumps({"event_type": "usage", "slide_count": 2}) + "\n")

        # Load events for specific date
        events = load_events(tmp_path, "usage", date="2026-01-20")
        assert len(events) == 1
        assert events[0]["slide_count"] == 1

    def test_load_events_empty_lines(self, tmp_path):
        """Test loading events with empty lines in file."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        file_path = daily_dir / "failure_2026-01-20.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"error_type": "Error1"}) + "\n")
            f.write("\n")  # Empty line
            f.write(json.dumps({"error_type": "Error2"}) + "\n")
            f.write("  \n")  # Whitespace line

        events = load_events(tmp_path, "failure", date="2026-01-20")
        assert len(events) == 2


class TestIsReportEmpty:
    """Tests for is_report_empty function."""

    def test_all_empty(self):
        """Test with all empty lists."""
        assert is_report_empty([], [], [], []) is True

    def test_has_sessions(self):
        """Test with sessions data."""
        sessions = [{"event_type": "app_start"}]
        assert is_report_empty(sessions, [], [], []) is False

    def test_has_usage(self):
        """Test with usage data."""
        usage = [{"event_type": "analysis_start"}]
        assert is_report_empty([], usage, [], []) is False

    def test_has_resources(self):
        """Test with resource data."""
        resources = [{"tile_count": 1000}]
        assert is_report_empty([], [], resources, []) is False

    def test_has_failures(self):
        """Test with failure data."""
        failures = [{"error_type": "ValueError"}]
        assert is_report_empty([], [], [], failures) is False

    def test_multiple_data_types(self):
        """Test with multiple data types."""
        sessions = [{"event_type": "app_start"}]
        failures = [{"error_type": "ValueError"}]
        assert is_report_empty(sessions, [], [], failures) is False


class TestGenerateTextReport:
    """Tests for generate_text_report function."""

    def test_empty_report(self, tmp_path):
        """Test generating report with no data."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        report = generate_text_report(tmp_path)
        assert "MOSAIC TELEMETRY REPORT" in report
        assert "NO FAILURES" in report
        assert "===" in report

    def test_report_with_sessions(self, tmp_path):
        """Test report with session data."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        sessions = [
            {
                "event_type": "app_shutdown",
                "uptime_sec": 3600,
                "analysis_time_sec": 1800,
                "analysis_count": 5,
                "hourly_rate": 0.40,
            }
        ]

        file_path = daily_dir / "session_2026-01-20.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for session in sessions:
                f.write(json.dumps(session) + "\n")

        report = generate_text_report(tmp_path)
        assert "COST SUMMARY" in report
        assert "Total uptime: 1.00 hours" in report
        assert "Estimated cost: $0.40" in report
        assert "App sessions: 1" in report

    def test_report_with_heartbeats(self, tmp_path):
        """Test report with running sessions (heartbeats)."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        heartbeats = [
            {
                "event_type": "heartbeat",
                "app_start_time": "2026-01-20T10:00:00Z",
                "uptime_sec": 7200,
                "analysis_time_sec": 3600,
                "analysis_count": 3,
                "hourly_rate": 0.40,
            }
        ]

        file_path = daily_dir / "session_2026-01-20.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for hb in heartbeats:
                f.write(json.dumps(hb) + "\n")

        report = generate_text_report(tmp_path)
        assert "COST SUMMARY" in report
        assert "Running sessions: 1" in report

    def test_report_with_usage(self, tmp_path):
        """Test report with usage data."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        usage = [
            {
                "event_type": "analysis_start",
                "slide_count": 3,
                "session_hash": "session1",
                "site_type": "Primary",
                "seg_config": "Resection",
            },
            {
                "event_type": "analysis_complete",
                "success": True,
                "duration_sec": 300,
            },
        ]

        file_path = daily_dir / "usage_2026-01-20.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for event in usage:
                f.write(json.dumps(event) + "\n")

        report = generate_text_report(tmp_path)
        assert "USAGE SUMMARY" in report
        assert "Analyses started: 1" in report
        assert "Analyses completed: 1" in report
        assert "Total slides processed: 3" in report
        assert "Primary: 1" in report
        assert "Resection: 1" in report

    def test_report_with_failures(self, tmp_path):
        """Test report with failure data."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        failures = [
            {
                "error_type": "ValueError",
                "error_message": "Invalid input parameter",
                "error_stage": "preprocessing",
            },
            {
                "error_type": "RuntimeError",
                "error_message": "GPU out of memory",
                "error_stage": "inference",
            },
            {
                "error_type": "ValueError",
                "error_message": "Another ValueError",
                "error_stage": "postprocessing",
            },
        ]

        file_path = daily_dir / "failure_2026-01-20.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for failure in failures:
                f.write(json.dumps(failure) + "\n")

        report = generate_text_report(tmp_path)
        assert "FAILURES (3)" in report
        assert "ValueError: 2" in report
        assert "RuntimeError: 1" in report
        assert "Recent failure messages:" in report

    def test_report_with_date_filter(self, tmp_path):
        """Test report with specific date filter."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        # Create files for two dates
        for date in ["2026-01-20", "2026-01-21"]:
            file_path = daily_dir / f"usage_{date}.jsonl"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps({"event_type": "analysis_start", "slide_count": 1})
                    + "\n"
                )

        report = generate_text_report(tmp_path, date="2026-01-20")
        assert "for 2026-01-20" in report

    def test_report_with_cached_analysis(self, tmp_path):
        """Test report shows cached analysis count when present."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        usage = [
            {
                "event_type": "analysis_start",
                "slide_count": 1,
                "session_hash": "s1",
                "site_type": "Primary",
                "seg_config": "TCGA",
            },
            {
                "event_type": "analysis_complete",
                "success": True,
                "duration_sec": 2.0,
                "slide_count": 1,
                "from_cache": True,
            },
            {
                "event_type": "analysis_start",
                "slide_count": 1,
                "session_hash": "s2",
                "site_type": "Primary",
                "seg_config": "TCGA",
            },
            {
                "event_type": "analysis_complete",
                "success": True,
                "duration_sec": 120.0,
                "slide_count": 1,
            },
        ]

        file_path = daily_dir / "usage_2026-01-20.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for event in usage:
                f.write(json.dumps(event) + "\n")

        report = generate_text_report(tmp_path)
        assert "Cached analyses: 1" in report
        # Average duration should exclude cached analysis (2s),
        # only include the fresh one (120s)
        assert "Average analysis duration: 120.0s" in report

    def test_report_without_cached_analysis(self, tmp_path):
        """Test report omits cache stats when no cache hits."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        usage = [
            {
                "event_type": "analysis_start",
                "slide_count": 1,
                "session_hash": "s1",
                "site_type": "Primary",
                "seg_config": "Biopsy",
            },
            {
                "event_type": "analysis_complete",
                "success": True,
                "duration_sec": 60.0,
                "slide_count": 1,
            },
        ]

        file_path = daily_dir / "usage_2026-01-20.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for event in usage:
                f.write(json.dumps(event) + "\n")

        report = generate_text_report(tmp_path)
        assert "Cached analyses" not in report

    def test_report_with_resources(self, tmp_path):
        """Test report with resource data."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        resources = [
            {
                "total_duration_sec": 7200,
                "tile_count": 50000,
                "peak_gpu_memory_gb": 12.5,
            }
        ]

        file_path = daily_dir / "resource_2026-01-20.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for resource in resources:
                f.write(json.dumps(resource) + "\n")

        report = generate_text_report(tmp_path)
        assert "RESOURCE SUMMARY" in report
        assert "Total slide processing time: 2.00 hours" in report
        assert "Total tiles processed: 50,000" in report
        assert "Peak GPU memory: 12.50 GB" in report


class TestGenerateHtmlReport:
    """Tests for generate_html_report function."""

    def test_html_structure(self, tmp_path):
        """Test basic HTML structure."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        report = generate_html_report(tmp_path)
        assert "<!DOCTYPE html>" in report
        assert "<html>" in report
        assert "</html>" in report
        assert "Mosaic Telemetry Report" in report

    def test_html_with_data(self, tmp_path):
        """Test HTML report with data."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        usage = [
            {"event_type": "analysis_start", "slide_count": 2, "session_hash": "s1"},
            {"event_type": "analysis_complete", "success": True},
        ]

        file_path = daily_dir / "usage_2026-01-20.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for event in usage:
                f.write(json.dumps(event) + "\n")

        report = generate_html_report(tmp_path)
        assert "<h2>Usage Summary</h2>" in report
        assert "<table>" in report
        assert "Analyses started" in report
        assert "<td>1</td>" in report  # 1 analysis started

    def test_html_with_cached_analysis(self, tmp_path):
        """Test HTML report shows cached analysis count."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        usage = [
            {
                "event_type": "analysis_start",
                "slide_count": 1,
                "session_hash": "s1",
            },
            {
                "event_type": "analysis_complete",
                "success": True,
                "duration_sec": 2.0,
                "slide_count": 1,
                "from_cache": True,
            },
        ]

        file_path = daily_dir / "usage_2026-01-20.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for event in usage:
                f.write(json.dumps(event) + "\n")

        report = generate_html_report(tmp_path)
        assert "Cached analyses" in report

    def test_html_with_failures(self, tmp_path):
        """Test HTML report with failures."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        failures = [
            {"error_type": "ValueError", "error_message": "Test error"},
        ]

        file_path = daily_dir / "failure_2026-01-20.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for failure in failures:
                f.write(json.dumps(failure) + "\n")

        report = generate_html_report(tmp_path)
        assert "<h2>Failures (1)</h2>" in report
        assert "ValueError" in report


class TestSendEmail:
    """Tests for send_email function."""

    @patch("telemetry_report.smtplib.SMTP")
    def test_send_text_email(self, mock_smtp):
        """Test sending text email."""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        report = "Test report content"
        send_email(report, "test@example.com", "Test Subject", format="text")

        mock_smtp.assert_called_once()
        mock_server.sendmail.assert_called_once()

    @patch("telemetry_report.smtplib.SMTP")
    def test_send_html_email(self, mock_smtp):
        """Test sending HTML email."""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        report = "<html><body>Test</body></html>"
        send_email(report, "test@example.com", "Test Subject", format="html")

        mock_smtp.assert_called_once()
        mock_server.sendmail.assert_called_once()

    @patch("telemetry_report.smtplib.SMTP")
    @patch.dict(
        "os.environ",
        {"SMTP_USER": "user", "SMTP_PASS": "pass"},
    )
    def test_send_email_with_auth(self, mock_smtp):
        """Test sending email with SMTP authentication."""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        report = "Test report"
        send_email(report, "test@example.com", "Test Subject")

        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("user", "pass")

    @patch("telemetry_report.smtplib.SMTP")
    @patch.dict(
        "os.environ",
        {
            "SMTP_HOST": "mail.example.com",
            "SMTP_PORT": "587",
            "SMTP_FROM": "noreply@example.com",
        },
    )
    def test_send_email_custom_config(self, mock_smtp):
        """Test sending email with custom SMTP configuration."""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        report = "Test report"
        send_email(report, "test@example.com", "Test Subject")

        mock_smtp.assert_called_once_with("mail.example.com", 587)


class TestIntegration:
    """Integration tests for complete report generation workflow."""

    def test_full_workflow_with_all_data(self, tmp_path):
        """Test complete workflow with all event types."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        # Create comprehensive test data
        sessions = [
            {
                "event_type": "app_shutdown",
                "uptime_sec": 3600,
                "analysis_time_sec": 1800,
                "analysis_count": 5,
                "hourly_rate": 0.40,
            }
        ]
        usage = [
            {
                "event_type": "analysis_start",
                "slide_count": 10,
                "session_hash": "test123",
                "site_type": "Primary",
                "seg_config": "Resection",
            },
            {
                "event_type": "analysis_complete",
                "success": True,
                "duration_sec": 600,
            },
        ]
        resources = [
            {
                "total_duration_sec": 1800,
                "tile_count": 25000,
                "peak_gpu_memory_gb": 8.5,
            }
        ]
        failures = [
            {
                "error_type": "ValueError",
                "error_message": "Invalid parameter",
                "error_stage": "preprocessing",
            }
        ]

        # Write all data
        for event_type, data in [
            ("session", sessions),
            ("usage", usage),
            ("resource", resources),
            ("failure", failures),
        ]:
            file_path = daily_dir / f"{event_type}_2026-01-20.jsonl"
            with open(file_path, "w", encoding="utf-8") as f:
                for event in data:
                    f.write(json.dumps(event) + "\n")

        # Test text report
        text_report = generate_text_report(tmp_path)
        assert "COST SUMMARY" in text_report
        assert "USAGE SUMMARY" in text_report
        assert "RESOURCE SUMMARY" in text_report
        assert "FAILURES (1)" in text_report

        # Test HTML report
        html_report = generate_html_report(tmp_path)
        assert "<h2>Cost Summary</h2>" in html_report
        assert "<h2>Usage Summary</h2>" in html_report
        assert "<h2>Failures (1)</h2>" in html_report

        # Verify not empty
        loaded_sessions = load_events(tmp_path, "session", "2026-01-20")
        loaded_usage = load_events(tmp_path, "usage", "2026-01-20")
        loaded_resources = load_events(tmp_path, "resource", "2026-01-20")
        loaded_failures = load_events(tmp_path, "failure", "2026-01-20")

        assert not is_report_empty(
            loaded_sessions, loaded_usage, loaded_resources, loaded_failures
        )

    def test_skip_empty_workflow(self, tmp_path):
        """Test workflow for skip-empty feature."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        # Create empty directory structure
        sessions = load_events(tmp_path, "session")
        usage = load_events(tmp_path, "usage")
        resources = load_events(tmp_path, "resource")
        failures = load_events(tmp_path, "failure")

        # Should be empty
        assert is_report_empty(sessions, usage, resources, failures) is True

        # Add one event
        file_path = daily_dir / "usage_2026-01-20.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps({"event_type": "analysis_start", "slide_count": 1}) + "\n"
            )

        # Should no longer be empty
        usage = load_events(tmp_path, "usage")
        assert is_report_empty([], usage, [], []) is False


class TestCLI:
    """Tests for command-line interface."""

    def test_main_with_skip_empty_no_data(self, tmp_path, capsys):
        """Test main function with --skip-empty and no data."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        # Mock sys.argv
        with patch("sys.argv", ["telemetry_report.py", str(tmp_path), "--skip-empty"]):
            with pytest.raises(SystemExit) as exc_info:
                from telemetry_report import main

                main()

            # Should exit with code 0 (success, but skipped)
            assert exc_info.value.code == 0

        # Check output
        captured = capsys.readouterr()
        assert "Skipping empty report" in captured.out

    def test_main_with_skip_empty_with_data(self, tmp_path, capsys):
        """Test main function with --skip-empty and data present."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        # Add some data
        file_path = daily_dir / "usage_2026-01-20.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps({"event_type": "analysis_start", "slide_count": 1}) + "\n"
            )

        # Mock sys.argv
        with patch("sys.argv", ["telemetry_report.py", str(tmp_path), "--skip-empty"]):
            from telemetry_report import main

            main()

        # Check that report was generated
        captured = capsys.readouterr()
        assert "MOSAIC TELEMETRY REPORT" in captured.out
        assert "USAGE SUMMARY" in captured.out

    def test_main_without_skip_empty(self, tmp_path, capsys):
        """Test main function without --skip-empty generates report even if empty."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        # Mock sys.argv
        with patch("sys.argv", ["telemetry_report.py", str(tmp_path)]):
            from telemetry_report import main

            main()

        # Check that report was generated even though it's empty
        captured = capsys.readouterr()
        assert "MOSAIC TELEMETRY REPORT" in captured.out
        assert "NO FAILURES" in captured.out

    def test_main_with_date_filter(self, tmp_path, capsys):
        """Test main function with date filter."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        # Create data for specific date
        file_path = daily_dir / "usage_2026-01-20.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps({"event_type": "analysis_start", "slide_count": 5}) + "\n"
            )

        # Mock sys.argv
        with patch(
            "sys.argv", ["telemetry_report.py", str(tmp_path), "--date", "2026-01-20"]
        ):
            from telemetry_report import main

            main()

        captured = capsys.readouterr()
        assert "for 2026-01-20" in captured.out

    @patch("telemetry_report.send_email")
    def test_main_with_email(self, mock_send_email, tmp_path, capsys):
        """Test main function with email option."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        # Add some data
        file_path = daily_dir / "usage_2026-01-20.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps({"event_type": "analysis_start", "slide_count": 1}) + "\n"
            )

        # Mock sys.argv
        with patch(
            "sys.argv",
            [
                "telemetry_report.py",
                str(tmp_path),
                "--email",
                "test@example.com",
            ],
        ):
            from telemetry_report import main

            main()

        # Verify email was sent
        mock_send_email.assert_called_once()
        captured = capsys.readouterr()
        assert "Report sent to test@example.com" in captured.out

    @patch("telemetry_report.send_email")
    def test_main_with_skip_empty_and_email(self, mock_send_email, tmp_path, capsys):
        """Test that email is not sent when report is empty with --skip-empty."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        # Mock sys.argv
        with patch(
            "sys.argv",
            [
                "telemetry_report.py",
                str(tmp_path),
                "--email",
                "test@example.com",
                "--skip-empty",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                from telemetry_report import main

                main()

            assert exc_info.value.code == 0

        # Verify email was NOT sent
        mock_send_email.assert_not_called()
        captured = capsys.readouterr()
        assert "Skipping empty report" in captured.out

    def test_main_html_format(self, tmp_path, capsys):
        """Test main function with HTML format."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()

        # Add some data
        file_path = daily_dir / "usage_2026-01-20.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps({"event_type": "analysis_start", "slide_count": 1}) + "\n"
            )

        # Mock sys.argv
        with patch(
            "sys.argv",
            ["telemetry_report.py", str(tmp_path), "--format", "html"],
        ):
            from telemetry_report import main

            main()

        captured = capsys.readouterr()
        assert "<!DOCTYPE html>" in captured.out
        assert "<html>" in captured.out
