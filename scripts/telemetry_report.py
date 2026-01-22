#!/usr/bin/env python3
"""Generate usage reports from telemetry data.

This script analyzes Mosaic telemetry data and generates reports for:
- Cost tracking (app uptime and estimated costs)
- Usage summary (analyses, slides, sessions)
- Failure analysis

Usage:
    # Full report (all time)
    python scripts/telemetry_report.py /path/to/telemetry

    # Daily report for yesterday (cron-friendly)
    python scripts/telemetry_report.py /path/to/telemetry --daily

    # Daily report for specific date
    python scripts/telemetry_report.py /path/to/telemetry --date 2026-01-20

    # Email output (pipe to sendmail or use with cron)
    python scripts/telemetry_report.py /path/to/telemetry --daily --email user@example.com

    # HTML format for email
    python scripts/telemetry_report.py /path/to/telemetry --daily --format html

Example cron entry (daily report at 8am):
    0 8 * * * python /app/scripts/telemetry_report.py /data/telemetry --daily --email team@example.com
"""

import argparse
import json
import os
import smtplib
import sys
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

DEFAULT_HOURLY_RATE = 0.40


def load_events(
    telemetry_dir: Path, event_type: str, date: Optional[str] = None
) -> list:
    """Load events from JSONL files.

    Args:
        telemetry_dir: Base telemetry directory
        event_type: Type of event ("session", "usage", "resource", "failure")
        date: Optional date filter in YYYY-MM-DD format

    Returns:
        List of event dictionaries
    """
    events = []
    daily_dir = telemetry_dir / "daily"

    if not daily_dir.exists():
        return events

    if date:
        # Load specific date file
        file_path = daily_dir / f"{event_type}_{date}.jsonl"
        if file_path.exists():
            with open(file_path, encoding="utf-8") as fp:
                for line in fp:
                    if line.strip():
                        events.append(json.loads(line))
    else:
        # Load all files
        for f in daily_dir.glob(f"{event_type}_*.jsonl"):
            with open(f, encoding="utf-8") as fp:
                for line in fp:
                    if line.strip():
                        events.append(json.loads(line))

    return events


def generate_text_report(
    telemetry_dir: Path, date: Optional[str] = None
) -> str:
    """Generate plain text report.

    Args:
        telemetry_dir: Base telemetry directory
        date: Optional date filter

    Returns:
        Report as string
    """
    sessions = load_events(telemetry_dir, "session", date)
    usage = load_events(telemetry_dir, "usage", date)
    resources = load_events(telemetry_dir, "resource", date)
    failures = load_events(telemetry_dir, "failure", date)

    lines = []
    date_label = f" for {date}" if date else " (All Time)"

    lines.append("=" * 60)
    lines.append(f"MOSAIC TELEMETRY REPORT{date_label}")
    lines.append("=" * 60)
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z")
    lines.append("")

    # Cost summary from session events
    if sessions:
        shutdowns = [s for s in sessions if s.get("event_type") == "app_shutdown"]
        if shutdowns:
            total_uptime_sec = sum(s.get("uptime_sec", 0) for s in shutdowns)
            total_uptime_hrs = total_uptime_sec / 3600
            total_analysis_sec = sum(s.get("analysis_time_sec", 0) for s in shutdowns)
            total_analysis_hrs = total_analysis_sec / 3600
            total_idle_hrs = total_uptime_hrs - total_analysis_hrs
            hourly_rate = shutdowns[0].get("hourly_rate", DEFAULT_HOURLY_RATE)
            total_cost = total_uptime_hrs * hourly_rate
            analysis_count = sum(s.get("analysis_count", 0) for s in shutdowns)

            utilization = (
                (total_analysis_hrs / total_uptime_hrs * 100)
                if total_uptime_hrs > 0
                else 0
            )

            lines.append("=== COST SUMMARY ===")
            lines.append(f"App sessions: {len(shutdowns)}")
            lines.append(f"Total uptime: {total_uptime_hrs:.2f} hours")
            lines.append(
                f"  - Active analysis: {total_analysis_hrs:.2f} hrs ({utilization:.1f}%)"
            )
            lines.append(
                f"  - Idle time: {total_idle_hrs:.2f} hrs ({100-utilization:.1f}%)"
            )
            lines.append(f"Estimated cost: ${total_cost:.2f} (@ ${hourly_rate}/hr)")
            if analysis_count > 0:
                lines.append(f"Cost per analysis: ${total_cost / analysis_count:.2f}")
            lines.append("")

    # Usage summary
    if usage:
        starts = [u for u in usage if u.get("event_type") == "analysis_start"]
        completes = [u for u in usage if u.get("event_type") == "analysis_complete"]
        successful = [c for c in completes if c.get("success", False)]

        total_slides = sum(s.get("slide_count", 0) for s in starts)
        unique_sessions = len(set(u.get("session_hash") for u in usage if u.get("session_hash")))

        # Calculate average duration
        durations = [c.get("duration_sec", 0) for c in completes if c.get("duration_sec")]
        avg_duration = sum(durations) / len(durations) if durations else 0

        lines.append("=== USAGE SUMMARY ===")
        lines.append(f"Analyses started: {len(starts)}")
        lines.append(f"Analyses completed: {len(completes)}")
        lines.append(f"Successful analyses: {len(successful)}")
        lines.append(f"Total slides processed: {total_slides}")
        lines.append(f"Unique sessions: {unique_sessions}")
        if avg_duration > 0:
            lines.append(f"Average analysis duration: {avg_duration:.1f}s")
        lines.append("")

        # Breakdown by settings
        site_types = {}
        seg_configs = {}
        for s in starts:
            st = s.get("site_type", "Unknown")
            site_types[st] = site_types.get(st, 0) + 1
            sc = s.get("seg_config", "Unknown")
            seg_configs[sc] = seg_configs.get(sc, 0) + 1

        if site_types:
            lines.append("By site type:")
            for st, count in sorted(site_types.items(), key=lambda x: -x[1]):
                lines.append(f"  {st}: {count}")
            lines.append("")

        if seg_configs:
            lines.append("By segmentation config:")
            for sc, count in sorted(seg_configs.items(), key=lambda x: -x[1]):
                lines.append(f"  {sc}: {count}")
            lines.append("")

    # Resource summary
    if resources:
        total_duration = sum(r.get("total_duration_sec", 0) for r in resources)
        total_tiles = sum(r.get("tile_count", 0) for r in resources if r.get("tile_count"))
        peak_memory = max(
            (r.get("peak_gpu_memory_gb", 0) for r in resources), default=0
        )

        lines.append("=== RESOURCE SUMMARY ===")
        lines.append(f"Total slide processing time: {total_duration / 3600:.2f} hours")
        lines.append(f"Total tiles processed: {total_tiles:,}")
        if peak_memory > 0:
            lines.append(f"Peak GPU memory: {peak_memory:.2f} GB")
        lines.append("")

    # Failure summary
    if failures:
        lines.append(f"=== FAILURES ({len(failures)}) ===")
        error_counts = {}
        for f in failures:
            error_type = f.get("error_type", "Unknown")
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {error_type}: {count}")

        # Show recent failure messages
        lines.append("")
        lines.append("Recent failure messages:")
        for f in failures[-5:]:
            msg = f.get("error_message", "")[:100]
            stage = f.get("error_stage", "unknown")
            lines.append(f"  [{stage}] {msg}")
        lines.append("")
    else:
        lines.append("=== NO FAILURES ===")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def generate_html_report(
    telemetry_dir: Path, date: Optional[str] = None
) -> str:
    """Generate HTML report.

    Args:
        telemetry_dir: Base telemetry directory
        date: Optional date filter

    Returns:
        Report as HTML string
    """
    sessions = load_events(telemetry_dir, "session", date)
    usage = load_events(telemetry_dir, "usage", date)
    resources = load_events(telemetry_dir, "resource", date)
    failures = load_events(telemetry_dir, "failure", date)

    date_label = f" for {date}" if date else " (All Time)"

    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html><head>")
    html.append("<meta charset='utf-8'>")
    html.append(f"<title>Mosaic Telemetry Report{date_label}</title>")
    html.append("<style>")
    html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
    html.append("h1 { color: #2c3e50; }")
    html.append("h2 { color: #34495e; border-bottom: 1px solid #eee; }")
    html.append("table { border-collapse: collapse; margin: 10px 0; }")
    html.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
    html.append("th { background-color: #f5f5f5; }")
    html.append(".metric { font-size: 24px; font-weight: bold; color: #2980b9; }")
    html.append(".cost { color: #e74c3c; }")
    html.append(".success { color: #27ae60; }")
    html.append("</style>")
    html.append("</head><body>")

    html.append(f"<h1>Mosaic Telemetry Report{date_label}</h1>")
    html.append(f"<p>Generated: {datetime.utcnow().isoformat()}Z</p>")

    # Cost summary
    if sessions:
        shutdowns = [s for s in sessions if s.get("event_type") == "app_shutdown"]
        if shutdowns:
            total_uptime_sec = sum(s.get("uptime_sec", 0) for s in shutdowns)
            total_uptime_hrs = total_uptime_sec / 3600
            total_analysis_sec = sum(s.get("analysis_time_sec", 0) for s in shutdowns)
            total_analysis_hrs = total_analysis_sec / 3600
            hourly_rate = shutdowns[0].get("hourly_rate", DEFAULT_HOURLY_RATE)
            total_cost = total_uptime_hrs * hourly_rate
            analysis_count = sum(s.get("analysis_count", 0) for s in shutdowns)
            utilization = (
                (total_analysis_hrs / total_uptime_hrs * 100)
                if total_uptime_hrs > 0
                else 0
            )

            html.append("<h2>Cost Summary</h2>")
            html.append("<table>")
            html.append(f"<tr><td>App sessions</td><td>{len(shutdowns)}</td></tr>")
            html.append(
                f"<tr><td>Total uptime</td><td>{total_uptime_hrs:.2f} hours</td></tr>"
            )
            html.append(
                f"<tr><td>Active analysis time</td><td>{total_analysis_hrs:.2f} hours ({utilization:.1f}%)</td></tr>"
            )
            html.append(
                f"<tr><td>Estimated cost</td><td class='cost'>${total_cost:.2f}</td></tr>"
            )
            if analysis_count > 0:
                html.append(
                    f"<tr><td>Cost per analysis</td><td>${total_cost/analysis_count:.2f}</td></tr>"
                )
            html.append("</table>")

    # Usage summary
    if usage:
        starts = [u for u in usage if u.get("event_type") == "analysis_start"]
        completes = [u for u in usage if u.get("event_type") == "analysis_complete"]
        successful = [c for c in completes if c.get("success", False)]
        total_slides = sum(s.get("slide_count", 0) for s in starts)
        unique_sessions = len(set(u.get("session_hash") for u in usage if u.get("session_hash")))

        html.append("<h2>Usage Summary</h2>")
        html.append("<table>")
        html.append(f"<tr><td>Analyses started</td><td>{len(starts)}</td></tr>")
        html.append(f"<tr><td>Analyses completed</td><td>{len(completes)}</td></tr>")
        html.append(
            f"<tr><td>Successful analyses</td><td class='success'>{len(successful)}</td></tr>"
        )
        html.append(f"<tr><td>Total slides</td><td>{total_slides}</td></tr>")
        html.append(f"<tr><td>Unique sessions</td><td>{unique_sessions}</td></tr>")
        html.append("</table>")

    # Failures
    if failures:
        html.append(f"<h2>Failures ({len(failures)})</h2>")
        html.append("<table>")
        html.append("<tr><th>Error Type</th><th>Count</th></tr>")
        error_counts = {}
        for f in failures:
            error_type = f.get("error_type", "Unknown")
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1])[:10]:
            html.append(f"<tr><td>{error_type}</td><td>{count}</td></tr>")
        html.append("</table>")

    html.append("</body></html>")

    return "\n".join(html)


def send_email(report: str, to_email: str, subject: str, format: str = "text"):
    """Send report via email using SMTP.

    Args:
        report: Report content
        to_email: Recipient email address
        subject: Email subject
        format: "text" or "html"
    """
    from_email = os.environ.get("SMTP_FROM", "mosaic-telemetry@noreply.local")
    smtp_host = os.environ.get("SMTP_HOST", "localhost")
    smtp_port_env = os.environ.get("SMTP_PORT", "25")
    try:
        smtp_port = int(smtp_port_env)
    except ValueError:
        smtp_port = 25
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    if format == "html":
        msg.attach(MIMEText(report, "html"))
    else:
        msg.attach(MIMEText(report, "plain"))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        if smtp_user and smtp_pass:
            server.starttls()
            server.login(smtp_user, smtp_pass)
        server.sendmail(from_email, [to_email], msg.as_string())


def main():
    parser = argparse.ArgumentParser(
        description="Generate Mosaic telemetry reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "telemetry_dir",
        type=Path,
        nargs="?",
        default=Path("/tmp/mosaic_telemetry"),
        help="Telemetry directory (default: /tmp/mosaic_telemetry)",
    )
    parser.add_argument(
        "--daily",
        action="store_true",
        help="Report for yesterday only",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Report for specific date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--email",
        type=str,
        help="Send report to this email address",
    )
    parser.add_argument(
        "--format",
        choices=["text", "html"],
        default="text",
        help="Output format (default: text)",
    )
    args = parser.parse_args()

    if not args.telemetry_dir.exists():
        print(f"Telemetry directory not found: {args.telemetry_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine date filter
    date = args.date
    if args.daily and not date:
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Generate report
    if args.format == "html":
        report = generate_html_report(args.telemetry_dir, date=date)
    else:
        report = generate_text_report(args.telemetry_dir, date=date)

    # Output
    if args.email:
        subject = f"Mosaic Telemetry Report - {date or 'All Time'}"
        try:
            send_email(report, args.email, subject, args.format)
            print(f"Report sent to {args.email}")
        except Exception as e:
            print(f"Failed to send email: {e}", file=sys.stderr)
            print(report)  # Print report to stdout as fallback
            sys.exit(1)
    else:
        print(report)


if __name__ == "__main__":
    main()
