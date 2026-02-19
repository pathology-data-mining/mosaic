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

    # Detailed report with per-session breakdown
    python scripts/telemetry_report.py /path/to/telemetry --long

    # Email output (pipe to sendmail or use with cron)
    python scripts/telemetry_report.py /path/to/telemetry --daily --email user@example.com

    # Skip email if report is empty (useful for automated daily reports)
    python scripts/telemetry_report.py /path/to/telemetry --daily --email user@example.com --skip-empty

    # HTML format for email
    python scripts/telemetry_report.py /path/to/telemetry --daily --format html

    # Long format with HTML
    python scripts/telemetry_report.py /path/to/telemetry --long --format html

    # Pull data from HuggingFace Dataset repository
    python scripts/telemetry_report.py --hf-repo PDM-Group/mosaic-telemetry

    # Pull from HF and save to specific directory
    python scripts/telemetry_report.py /path/to/telemetry --hf-repo PDM-Group/mosaic-telemetry

Example cron entry (daily report at 8am, skip if empty):
    0 8 * * * python /app/scripts/telemetry_report.py /data/telemetry --daily --email team@example.com --skip-empty
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


def is_report_empty(
    sessions: list, usage: list, resources: list, failures: list
) -> bool:
    """Check if report would be empty (no meaningful data).

    Args:
        sessions: Session events
        usage: Usage events
        resources: Resource events
        failures: Failure events

    Returns:
        True if report is empty, False otherwise
    """
    # Check if there are any meaningful events
    has_sessions = bool(sessions)
    has_usage = bool(usage)
    has_resources = bool(resources)
    has_failures = bool(failures)

    return not (has_sessions or has_usage or has_resources or has_failures)


def generate_text_report(telemetry_dir: Path, date: Optional[str] = None, long: bool = False) -> str:
    """Generate plain text report.

    Args:
        telemetry_dir: Base telemetry directory
        date: Optional date filter
        long: Include detailed per-session breakdown

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

        # For running instances without shutdowns, use the latest heartbeat per session
        if not shutdowns:
            # Group heartbeats by app_start_time to identify unique sessions
            heartbeats = [s for s in sessions if s.get("event_type") == "heartbeat"]
            if heartbeats:
                # Get the latest heartbeat for each session (by app_start_time)
                sessions_by_start = {}
                for hb in heartbeats:
                    start_time = hb.get("app_start_time")
                    if start_time:
                        if start_time not in sessions_by_start or hb.get(
                            "uptime_sec", 0
                        ) > sessions_by_start[start_time].get("uptime_sec", 0):
                            sessions_by_start[start_time] = hb
                shutdowns = list(sessions_by_start.values())

        if shutdowns:
            total_uptime_sec = sum(s.get("uptime_sec", 0) for s in shutdowns)
            total_uptime_hrs = total_uptime_sec / 3600
            total_analysis_sec = sum(s.get("analysis_time_sec", 0) for s in shutdowns)
            total_analysis_hrs = total_analysis_sec / 3600
            total_idle_hrs = total_uptime_hrs - total_analysis_hrs
            # Use hourly_rate from data, fallback to DEFAULT if missing or zero
            hourly_rate = shutdowns[0].get("hourly_rate") or DEFAULT_HOURLY_RATE
            total_cost = total_uptime_hrs * hourly_rate
            analysis_count = sum(s.get("analysis_count", 0) for s in shutdowns)

            utilization = (
                (total_analysis_hrs / total_uptime_hrs * 100)
                if total_uptime_hrs > 0
                else 0
            )

            # Check if these are from running instances (heartbeats) vs completed (shutdowns)
            is_running = all(s.get("event_type") == "heartbeat" for s in shutdowns)
            session_label = (
                f"Running sessions: {len(shutdowns)}"
                if is_running
                else f"App sessions: {len(shutdowns)}"
            )

            lines.append("=== COST SUMMARY ===")
            lines.append(session_label)
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
        unique_sessions = len(
            set(u.get("session_hash") for u in usage if u.get("session_hash"))
        )

        # Cache hit tracking
        total_cached_slides = sum(
            c.get("cached_slide_count", 0) for c in completes if c.get("cached_slide_count")
        )
        fully_cached_analyses = [
            c for c in completes
            if c.get("cached_slide_count") and c.get("cached_slide_count") == c.get("slide_count", 0)
        ]
        fresh_analyses = [
            c for c in completes
            if not c.get("cached_slide_count") or c.get("cached_slide_count") < c.get("slide_count", 0)
        ]

        # Calculate average duration (excluding fully cached analyses)
        durations = [
            c.get("duration_sec", 0) for c in fresh_analyses if c.get("duration_sec")
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0

        lines.append("=== USAGE SUMMARY ===")
        lines.append(f"Analyses started: {len(starts)}")
        lines.append(f"Analyses completed: {len(completes)}")
        lines.append(f"Successful analyses: {len(successful)}")
        lines.append(f"Total slides processed: {total_slides}")
        if total_cached_slides > 0:
            lines.append(f"Cached slides: {total_cached_slides}")
            cache_rate = (total_cached_slides / total_slides * 100) if total_slides > 0 else 0
            lines.append(f"Cache hit rate: {cache_rate:.1f}%")
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

    # User summary (from usage events)
    all_events = usage + resources + failures
    logged_in_events = [e for e in all_events if e.get("is_logged_in")]
    anonymous_events = [e for e in all_events if not e.get("is_logged_in")]
    if logged_in_events or anonymous_events:
        lines.append("=== USER SUMMARY ===")
        lines.append(
            f"Logged-in users: {len(set(e.get('hf_username') for e in logged_in_events if e.get('hf_username')))}"
        )
        lines.append(
            f"Anonymous sessions: {len(set(e.get('session_hash') for e in anonymous_events if e.get('session_hash')))}"
        )
        lines.append("")

        # Per-user breakdown from analysis_start events
        user_starts = [
            u
            for u in usage
            if u.get("event_type") == "analysis_start" and u.get("hf_username")
        ]
        if user_starts:
            user_stats = {}
            for u in user_starts:
                name = u["hf_username"]
                if name not in user_stats:
                    user_stats[name] = {"analyses": 0, "slides": 0}
                user_stats[name]["analyses"] += 1
                user_stats[name]["slides"] += u.get("slide_count", 0)

            lines.append("By user:")
            for name, stats in sorted(
                user_stats.items(), key=lambda x: -x[1]["analyses"]
            ):
                lines.append(
                    f"  {name}: {stats['analyses']} analyses, {stats['slides']} slides"
                )
            lines.append("")

    # Resource summary
    if resources:
        total_duration = sum(r.get("total_duration_sec", 0) for r in resources)
        total_tiles = sum(
            r.get("tile_count", 0) for r in resources if r.get("tile_count")
        )
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

    # Long format: detailed per-session breakdown
    if long and sessions:
        lines.append("=== DETAILED SESSION BREAKDOWN ===")
        shutdowns = [s for s in sessions if s.get("event_type") == "app_shutdown"]
        heartbeats = [s for s in sessions if s.get("event_type") == "heartbeat"]
        
        # Group heartbeats by app_start_time for running sessions
        all_sessions = []
        
        if shutdowns:
            all_sessions.extend(shutdowns)
        
        if heartbeats:
            sessions_by_start = {}
            for hb in heartbeats:
                start_time = hb.get("app_start_time")
                if start_time:
                    if start_time not in sessions_by_start or hb.get("uptime_sec", 0) > sessions_by_start[start_time].get("uptime_sec", 0):
                        sessions_by_start[start_time] = hb
            all_sessions.extend(sessions_by_start.values())
        
        # Sort by timestamp
        all_sessions.sort(key=lambda x: x.get("timestamp", x.get("app_start_time", "")))
        
        for i, session in enumerate(all_sessions, 1):
            is_running = session.get("event_type") == "heartbeat"
            uptime_sec = session.get("uptime_sec", 0)
            uptime_hrs = uptime_sec / 3600
            analysis_sec = session.get("analysis_time_sec", 0)
            analysis_hrs = analysis_sec / 3600
            idle_hrs = uptime_hrs - analysis_hrs
            analysis_count = session.get("analysis_count", 0)
            hourly_rate = session.get("hourly_rate") or DEFAULT_HOURLY_RATE
            cost = uptime_hrs * hourly_rate
            utilization = (analysis_hrs / uptime_hrs * 100) if uptime_hrs > 0 else 0
            
            start_time = session.get("app_start_time", session.get("timestamp", "Unknown"))
            status = "Running" if is_running else "Completed"
            
            lines.append(f"\nSession {i} [{status}]:")
            lines.append(f"  Start time: {start_time}")
            lines.append(f"  Uptime: {uptime_hrs:.2f} hrs ({uptime_sec} sec)")
            lines.append(f"  Active analysis: {analysis_hrs:.2f} hrs ({utilization:.1f}%)")
            lines.append(f"  Idle time: {idle_hrs:.2f} hrs ({100-utilization:.1f}%)")
            lines.append(f"  Analyses completed: {analysis_count}")
            lines.append(f"  Cost: ${cost:.2f} (@ ${hourly_rate}/hr)")
            if analysis_count > 0:
                lines.append(f"  Cost per analysis: ${cost / analysis_count:.2f}")
        lines.append("")
    
    # Long format: detailed analysis breakdown
    if long and usage:
        starts = [u for u in usage if u.get("event_type") == "analysis_start"]
        completes = [u for u in usage if u.get("event_type") == "analysis_complete"]
        
        # Create a map of completes by timestamp proximity to starts
        lines.append("=== DETAILED ANALYSIS BREAKDOWN ===")
        
        # Sort starts by timestamp
        starts_sorted = sorted(starts, key=lambda x: x.get("timestamp", ""))
        completes_sorted = sorted(completes, key=lambda x: x.get("timestamp", ""))
        
        # Track which completes we've already matched
        used_completes = set()
        
        for i, start in enumerate(starts_sorted, 1):
            timestamp = start.get("timestamp", "Unknown")
            session_hash = start.get("session_hash", "Unknown")
            slide_count = start.get("slide_count", 0)
            site_type = start.get("site_type", "Unknown")
            seg_config = start.get("seg_config", "Unknown")
            hf_username = start.get("hf_username") or "Anonymous"
            
            # Try to find corresponding complete event (first unused one after this start)
            complete = None
            for j, c in enumerate(completes_sorted):
                if j not in used_completes and c.get("timestamp", "") >= timestamp:
                    complete = c
                    used_completes.add(j)
                    break
            
            lines.append(f"\nAnalysis {i}:")
            lines.append(f"  Timestamp: {timestamp}")
            lines.append(f"  User: {hf_username}")
            lines.append(f"  Session: {session_hash[:16]}...")
            lines.append(f"  Slides: {slide_count}")
            lines.append(f"  Site type: {site_type}")
            lines.append(f"  Segmentation: {seg_config}")
            
            if complete:
                success = complete.get("success", False)
                duration = complete.get("duration_sec", 0)
                cached_count = complete.get("cached_slide_count")
                status = "Success" if success else "Failed"
                
                lines.append(f"  Status: {status}")
                lines.append(f"  Duration: {duration:.1f}s")
                if cached_count and cached_count > 0:
                    lines.append(f"  Cached slides: {cached_count}/{slide_count}")
            else:
                lines.append(f"  Status: No completion event found")
        
        lines.append("")
    
    # Long format: detailed failure breakdown
    if long and failures:
        lines.append("=== DETAILED FAILURE BREAKDOWN ===")
        
        # Sort failures by timestamp (newest first)
        failures_sorted = sorted(failures, key=lambda x: x.get("timestamp", ""), reverse=True)
        
        for i, failure in enumerate(failures_sorted, 1):
            timestamp = failure.get("timestamp", "Unknown")
            error_type = failure.get("error_type", "Unknown")
            error_message = failure.get("error_message", "No message")
            error_stage = failure.get("error_stage", "Unknown")
            session_hash = failure.get("session_hash", "Unknown")
            hf_username = failure.get("hf_username", "Anonymous")
            slide_path = failure.get("slide_path", "N/A")
            stack_trace = failure.get("stack_trace", "")
            
            lines.append(f"\nFailure {i}:")
            lines.append(f"  Timestamp: {timestamp}")
            lines.append(f"  User: {hf_username}")
            lines.append(f"  Session: {session_hash[:16] if len(session_hash) > 16 else session_hash}...")
            lines.append(f"  Error type: {error_type}")
            lines.append(f"  Stage: {error_stage}")
            lines.append(f"  Slide: {slide_path}")
            lines.append(f"  Message: {error_message}")
            
            if stack_trace:
                lines.append(f"  Stack trace:")
                # Indent stack trace
                for line in stack_trace.split("\n")[:20]:  # Limit to 20 lines
                    lines.append(f"    {line}")
        
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def generate_html_report(telemetry_dir: Path, date: Optional[str] = None, long: bool = False) -> str:
    """Generate HTML report.

    Args:
        telemetry_dir: Base telemetry directory
        date: Optional date filter
        long: Include detailed per-session breakdown

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

        # For running instances without shutdowns, use the latest heartbeat per session
        if not shutdowns:
            heartbeats = [s for s in sessions if s.get("event_type") == "heartbeat"]
            if heartbeats:
                sessions_by_start = {}
                for hb in heartbeats:
                    start_time = hb.get("app_start_time")
                    if start_time:
                        if start_time not in sessions_by_start or hb.get(
                            "uptime_sec", 0
                        ) > sessions_by_start[start_time].get("uptime_sec", 0):
                            sessions_by_start[start_time] = hb
                shutdowns = list(sessions_by_start.values())

        if shutdowns:
            total_uptime_sec = sum(s.get("uptime_sec", 0) for s in shutdowns)
            total_uptime_hrs = total_uptime_sec / 3600
            total_analysis_sec = sum(s.get("analysis_time_sec", 0) for s in shutdowns)
            total_analysis_hrs = total_analysis_sec / 3600
            hourly_rate = shutdowns[0].get("hourly_rate") or DEFAULT_HOURLY_RATE
            total_cost = total_uptime_hrs * hourly_rate
            analysis_count = sum(s.get("analysis_count", 0) for s in shutdowns)
            utilization = (
                (total_analysis_hrs / total_uptime_hrs * 100)
                if total_uptime_hrs > 0
                else 0
            )

            is_running = all(s.get("event_type") == "heartbeat" for s in shutdowns)
            session_label = (
                f"Running sessions: {len(shutdowns)}"
                if is_running
                else f"App sessions: {len(shutdowns)}"
            )

            html.append("<h2>Cost Summary</h2>")
            html.append("<table>")
            html.append(
                f"<tr><td>{session_label.split(':')[0]}</td><td>{len(shutdowns)}</td></tr>"
            )
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
        unique_sessions = len(
            set(u.get("session_hash") for u in usage if u.get("session_hash"))
        )

        # Cache hit tracking
        total_cached_slides = sum(
            c.get("cached_slide_count", 0) for c in completes if c.get("cached_slide_count")
        )

        html.append("<h2>Usage Summary</h2>")
        html.append("<table>")
        html.append(f"<tr><td>Analyses started</td><td>{len(starts)}</td></tr>")
        html.append(f"<tr><td>Analyses completed</td><td>{len(completes)}</td></tr>")
        html.append(
            f"<tr><td>Successful analyses</td><td class='success'>{len(successful)}</td></tr>"
        )
        html.append(f"<tr><td>Total slides</td><td>{total_slides}</td></tr>")
        if total_cached_slides > 0:
            html.append(
                f"<tr><td>Cached slides</td><td>{total_cached_slides}</td></tr>"
            )
            cache_rate = (total_cached_slides / total_slides * 100) if total_slides > 0 else 0
            html.append(
                f"<tr><td>Cache hit rate</td><td>{cache_rate:.1f}%</td></tr>"
            )
        html.append(f"<tr><td>Unique sessions</td><td>{unique_sessions}</td></tr>")
        html.append("</table>")

    # User summary
    all_events = usage + resources + failures
    logged_in_events = [e for e in all_events if e.get("is_logged_in")]
    anonymous_events = [e for e in all_events if not e.get("is_logged_in")]
    if logged_in_events or anonymous_events:
        unique_users = set(
            e.get("hf_username") for e in logged_in_events if e.get("hf_username")
        )
        anon_sessions = set(
            e.get("session_hash") for e in anonymous_events if e.get("session_hash")
        )

        html.append("<h2>User Summary</h2>")
        html.append("<table>")
        html.append(f"<tr><td>Logged-in users</td><td>{len(unique_users)}</td></tr>")
        html.append(
            f"<tr><td>Anonymous sessions</td><td>{len(anon_sessions)}</td></tr>"
        )
        html.append("</table>")

        # Per-user breakdown
        user_starts = [
            u
            for u in usage
            if u.get("event_type") == "analysis_start" and u.get("hf_username")
        ]
        if user_starts:
            user_stats = {}
            for u in user_starts:
                name = u["hf_username"]
                if name not in user_stats:
                    user_stats[name] = {"analyses": 0, "slides": 0}
                user_stats[name]["analyses"] += 1
                user_stats[name]["slides"] += u.get("slide_count", 0)

            html.append("<table>")
            html.append("<tr><th>User</th><th>Analyses</th><th>Slides</th></tr>")
            for name, stats in sorted(
                user_stats.items(), key=lambda x: -x[1]["analyses"]
            ):
                html.append(
                    f"<tr><td>{name}</td><td>{stats['analyses']}</td><td>{stats['slides']}</td></tr>"
                )
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

    # Long format: detailed session breakdown
    if long and sessions:
        html.append("<h2>Detailed Session Breakdown</h2>")
        shutdowns = [s for s in sessions if s.get("event_type") == "app_shutdown"]
        heartbeats = [s for s in sessions if s.get("event_type") == "heartbeat"]
        
        all_sessions = []
        if shutdowns:
            all_sessions.extend(shutdowns)
        
        if heartbeats:
            sessions_by_start = {}
            for hb in heartbeats:
                start_time = hb.get("app_start_time")
                if start_time:
                    if start_time not in sessions_by_start or hb.get("uptime_sec", 0) > sessions_by_start[start_time].get("uptime_sec", 0):
                        sessions_by_start[start_time] = hb
            all_sessions.extend(sessions_by_start.values())
        
        all_sessions.sort(key=lambda x: x.get("timestamp", x.get("app_start_time", "")))
        
        html.append("<table>")
        html.append("<tr><th>#</th><th>Status</th><th>Start Time</th><th>Uptime</th><th>Active</th><th>Utilization</th><th>Analyses</th><th>Cost</th></tr>")
        
        for i, session in enumerate(all_sessions, 1):
            is_running = session.get("event_type") == "heartbeat"
            uptime_sec = session.get("uptime_sec", 0)
            uptime_hrs = uptime_sec / 3600
            analysis_sec = session.get("analysis_time_sec", 0)
            analysis_hrs = analysis_sec / 3600
            analysis_count = session.get("analysis_count", 0)
            hourly_rate = session.get("hourly_rate") or DEFAULT_HOURLY_RATE
            cost = uptime_hrs * hourly_rate
            utilization = (analysis_hrs / uptime_hrs * 100) if uptime_hrs > 0 else 0
            
            start_time = session.get("app_start_time", session.get("timestamp", "Unknown"))
            status = "Running" if is_running else "Completed"
            status_class = "success" if not is_running else ""
            
            html.append(f"<tr>")
            html.append(f"<td>{i}</td>")
            html.append(f"<td class='{status_class}'>{status}</td>")
            html.append(f"<td>{start_time}</td>")
            html.append(f"<td>{uptime_hrs:.2f}h</td>")
            html.append(f"<td>{analysis_hrs:.2f}h</td>")
            html.append(f"<td>{utilization:.1f}%</td>")
            html.append(f"<td>{analysis_count}</td>")
            html.append(f"<td class='cost'>${cost:.2f}</td>")
            html.append(f"</tr>")
        
        html.append("</table>")
    
    # Long format: detailed analysis breakdown
    if long and usage:
        starts = [u for u in usage if u.get("event_type") == "analysis_start"]
        completes = [u for u in usage if u.get("event_type") == "analysis_complete"]
        
        if starts:
            html.append("<h2>Detailed Analysis Breakdown</h2>")
            starts_sorted = sorted(starts, key=lambda x: x.get("timestamp", ""))
            completes_sorted = sorted(completes, key=lambda x: x.get("timestamp", ""))
            
            html.append("<table>")
            html.append("<tr><th>#</th><th>Timestamp</th><th>User</th><th>Session</th><th>Slides</th><th>Site Type</th><th>Status</th><th>Duration</th><th>Cached</th></tr>")
            
            # Track which completes we've already matched
            used_completes = set()
            
            for i, start in enumerate(starts_sorted, 1):
                timestamp = start.get("timestamp", "Unknown")
                session_hash = start.get("session_hash", "Unknown")
                slide_count = start.get("slide_count", 0)
                site_type = start.get("site_type", "Unknown")
                hf_username = start.get("hf_username") or "Anonymous"
                
                # Try to find corresponding complete event (first unused one after this start)
                complete = None
                for j, c in enumerate(completes_sorted):
                    if j not in used_completes and c.get("timestamp", "") >= timestamp:
                        complete = c
                        used_completes.add(j)
                        break
                
                status = "N/A"
                duration = "N/A"
                cached_info = "N/A"
                status_class = ""
                
                if complete:
                    success = complete.get("success", False)
                    duration_sec = complete.get("duration_sec", 0)
                    cached_count = complete.get("cached_slide_count")
                    status = "Success" if success else "Failed"
                    status_class = "success" if success else "cost"
                    duration = f"{duration_sec:.1f}s"
                    if cached_count and cached_count > 0:
                        cached_info = f"{cached_count}/{slide_count}"
                
                html.append(f"<tr>")
                html.append(f"<td>{i}</td>")
                html.append(f"<td>{timestamp}</td>")
                html.append(f"<td>{hf_username}</td>")
                html.append(f"<td>{session_hash[:12]}...</td>")
                html.append(f"<td>{slide_count}</td>")
                html.append(f"<td>{site_type}</td>")
                html.append(f"<td class='{status_class}'>{status}</td>")
                html.append(f"<td>{duration}</td>")
                html.append(f"<td>{cached_info}</td>")
                html.append(f"</tr>")
            
            html.append("</table>")
    
    # Long format: detailed failure breakdown
    if long and failures:
        html.append("<h2>Detailed Failure Breakdown</h2>")
        failures_sorted = sorted(failures, key=lambda x: x.get("timestamp", ""), reverse=True)
        
        html.append("<table>")
        html.append("<tr><th>#</th><th>Timestamp</th><th>User</th><th>Error Type</th><th>Stage</th><th>Message</th></tr>")
        
        for i, failure in enumerate(failures_sorted, 1):
            timestamp = failure.get("timestamp", "Unknown")
            error_type = failure.get("error_type", "Unknown")
            error_message = failure.get("error_message", "No message")
            error_stage = failure.get("error_stage", "Unknown")
            hf_username = failure.get("hf_username", "Anonymous")
            
            # Truncate long messages for table
            if len(error_message) > 100:
                error_message_display = error_message[:97] + "..."
            else:
                error_message_display = error_message
            
            html.append(f"<tr>")
            html.append(f"<td>{i}</td>")
            html.append(f"<td>{timestamp}</td>")
            html.append(f"<td>{hf_username}</td>")
            html.append(f"<td>{error_type}</td>")
            html.append(f"<td>{error_stage}</td>")
            html.append(f"<td title='{error_message}'>{error_message_display}</td>")
            html.append(f"</tr>")
        
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


def download_from_hf(repo_id: str, telemetry_dir: Path) -> bool:
    """Download telemetry data from HuggingFace Dataset repository.

    Args:
        repo_id: HuggingFace Dataset repository ID
        telemetry_dir: Local directory to store downloaded files

    Returns:
        True if download was successful, False otherwise
    """
    try:
        from mosaic.telemetry.storage import TelemetryStorage
    except ImportError:
        # Fallback for standalone usage without mosaic installed
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError:
            print(
                "huggingface_hub not installed. Install with: pip install huggingface-hub",
                file=sys.stderr,
            )
            return False

        api = HfApi()
        daily_dir = telemetry_dir / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)

        try:
            files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        except Exception as e:
            print(f"Failed to list files in {repo_id}: {e}", file=sys.stderr)
            return False

        jsonl_files = [
            f for f in files if f.startswith("daily/") and f.endswith(".jsonl")
        ]
        if not jsonl_files:
            print(f"No telemetry files found in {repo_id}", file=sys.stderr)
            return False

        downloaded = 0
        for remote_path in jsonl_files:
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_path,
                    repo_type="dataset",
                )
                filename = os.path.basename(remote_path)
                target_path = daily_dir / filename

                with open(local_path, "r", encoding="utf-8") as f:
                    remote_content = f.read()

                if target_path.exists():
                    with open(target_path, "r", encoding="utf-8") as f:
                        local_content = f.read()
                    local_lines = (
                        set(local_content.strip().split("\n"))
                        if local_content.strip()
                        else set()
                    )
                    remote_lines = (
                        remote_content.strip().split("\n")
                        if remote_content.strip()
                        else []
                    )
                    new_lines = [
                        line
                        for line in remote_lines
                        if line and line not in local_lines
                    ]
                    if new_lines:
                        with open(target_path, "a", encoding="utf-8") as f:
                            for line in new_lines:
                                f.write(line + "\n")
                        print(f"Merged {len(new_lines)} new events into {filename}")
                else:
                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(remote_content)
                    print(f"Downloaded: {filename}")
                downloaded += 1
            except Exception as e:
                print(f"Failed to download {remote_path}: {e}", file=sys.stderr)

        return downloaded > 0

    # Use TelemetryStorage if mosaic is available
    storage = TelemetryStorage(telemetry_dir)
    return storage.download_from_hf_dataset(repo_id)


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
    parser.add_argument(
        "--hf-repo",
        type=str,
        help="HuggingFace Dataset repository to pull telemetry from (e.g., PDM-Group/mosaic-telemetry)",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip sending email if report has no data (useful for automated daily reports)",
    )
    parser.add_argument(
        "--long",
        action="store_true",
        help="Include detailed per-session breakdown in report",
    )
    args = parser.parse_args()

    # If HF repo specified, download to a clean temp directory
    if args.hf_repo:
        import tempfile

        # Use a clean temp directory to avoid mixing with local data
        temp_dir = Path(tempfile.mkdtemp(prefix="mosaic_telemetry_"))
        print(f"Downloading telemetry from {args.hf_repo}...")
        if not download_from_hf(args.hf_repo, temp_dir):
            print(
                f"ERROR: Failed to download telemetry data from {args.hf_repo}",
                file=sys.stderr,
            )
            print(
                "Please check HF_TOKEN is set and you have access to the repository",
                file=sys.stderr,
            )
            sys.exit(1)
        # Use the temp directory for report generation
        args.telemetry_dir = temp_dir

    if not args.telemetry_dir.exists():
        print(f"Telemetry directory not found: {args.telemetry_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine date filter
    date = args.date
    if args.daily and not date:
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Check if report would be empty before generating
    if args.skip_empty:
        sessions = load_events(args.telemetry_dir, "session", date)
        usage = load_events(args.telemetry_dir, "usage", date)
        resources = load_events(args.telemetry_dir, "resource", date)
        failures = load_events(args.telemetry_dir, "failure", date)

        if is_report_empty(sessions, usage, resources, failures):
            print(f"Skipping empty report for {date or 'all time'}")
            sys.exit(0)

    # Generate report
    if args.format == "html":
        report = generate_html_report(args.telemetry_dir, date=date, long=args.long)
    else:
        report = generate_text_report(args.telemetry_dir, date=date, long=args.long)

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
