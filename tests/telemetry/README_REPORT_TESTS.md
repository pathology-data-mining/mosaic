# Telemetry Report Tests

This document describes the test coverage for the telemetry report generation script (`scripts/telemetry_report.py`).

## Test File

`tests/telemetry/test_report.py` - Comprehensive test suite with 34 tests covering all major functionality.

## Test Coverage

### 1. TestLoadEvents (5 tests)
Tests for loading telemetry events from JSONL files:
- `test_load_events_no_directory` - Handles missing directories gracefully
- `test_load_events_empty_directory` - Returns empty list for empty directories
- `test_load_events_all_files` - Loads all events without date filter
- `test_load_events_specific_date` - Filters events by specific date
- `test_load_events_empty_lines` - Handles empty/whitespace lines in files

### 2. TestIsReportEmpty (6 tests)
Tests for the `is_report_empty()` function used with `--skip-empty`:
- `test_all_empty` - Returns True when all event lists are empty
- `test_has_sessions` - Returns False when session data exists
- `test_has_usage` - Returns False when usage data exists
- `test_has_resources` - Returns False when resource data exists
- `test_has_failures` - Returns False when failure data exists
- `test_multiple_data_types` - Returns False when multiple data types exist

### 3. TestGenerateTextReport (8 tests)
Tests for plain text report generation:
- `test_empty_report` - Generates minimal report with no data
- `test_report_with_sessions` - Includes cost summary from session data
- `test_report_with_heartbeats` - Handles running sessions (heartbeats)
- `test_report_with_usage` - Includes usage summary and breakdowns
- `test_report_with_failures` - Shows failure counts and messages
- `test_report_with_date_filter` - Filters report by date
- `test_report_with_resources` - Includes resource utilization metrics

### 4. TestGenerateHtmlReport (3 tests)
Tests for HTML report generation:
- `test_html_structure` - Verifies valid HTML structure
- `test_html_with_data` - Generates HTML tables with data
- `test_html_with_failures` - Includes failure information in HTML

### 5. TestSendEmail (4 tests)
Tests for email sending functionality:
- `test_send_text_email` - Sends plain text emails
- `test_send_html_email` - Sends HTML emails
- `test_send_email_with_auth` - Uses SMTP authentication when configured
- `test_send_email_custom_config` - Respects custom SMTP settings

### 6. TestIntegration (2 tests)
End-to-end integration tests:
- `test_full_workflow_with_all_data` - Complete workflow with all event types
- `test_skip_empty_workflow` - Validates skip-empty feature behavior

### 7. TestCLI (8 tests)
Command-line interface tests:
- `test_main_with_skip_empty_no_data` - Exits early when no data and `--skip-empty`
- `test_main_with_skip_empty_with_data` - Generates report when data exists
- `test_main_without_skip_empty` - Always generates report (even if empty)
- `test_main_with_date_filter` - Filters report by date via CLI
- `test_main_with_email` - Sends email when `--email` specified
- `test_main_with_skip_empty_and_email` - Does NOT send email when report is empty with `--skip-empty`
- `test_main_html_format` - Generates HTML output when `--format html`

## Running the Tests

```bash
# Run just the report tests
make test-specific TEST=tests/telemetry/test_report.py

# Run all telemetry tests
make test-specific TEST=tests/telemetry/

# Run full test suite
make test
```

## Key Features Tested

1. **Event Loading**: Loading from JSONL files, filtering by date, handling malformed data
2. **Report Generation**: Text and HTML formats, all event types, date filtering
3. **Email Sending**: SMTP with/without auth, custom configuration
4. **Skip Empty**: Core feature that prevents sending empty reports
5. **CLI Integration**: All command-line flags and combinations
6. **Error Handling**: Graceful handling of missing files, empty data, etc.

## Test Data

All tests use `tmp_path` fixtures (pytest's temporary directory) to avoid:
- Interfering with real telemetry data
- Leaving test artifacts in the repository
- Race conditions in parallel test execution

## Mocking

The tests use `unittest.mock` to:
- Mock SMTP server for email tests (no actual emails sent)
- Mock `sys.argv` for CLI tests
- Mock file system operations where needed

## Coverage

These tests provide comprehensive coverage of:
- ✅ Event loading logic
- ✅ Empty report detection
- ✅ Text report generation
- ✅ HTML report generation  
- ✅ Email sending
- ✅ CLI argument parsing
- ✅ Integration workflows
- ✅ Edge cases and error conditions

## Notes

- Tests run quickly (~0.7 seconds for all 34 tests)
- No external dependencies required (mocked SMTP, file system)
- Can run in CI/CD without network access
- Temporary directories automatically cleaned up after tests
