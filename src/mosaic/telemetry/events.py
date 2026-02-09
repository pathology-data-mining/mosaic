"""Event dataclasses for Mosaic telemetry.

This module defines the event types that are logged by the telemetry system:
- AppSessionEvent: Tracks app wake/sleep for billing
- UsageEvent: Tracks analysis requests
- ResourceEvent: Tracks compute resources per analysis
- FailureEvent: Tracks errors
"""

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Literal


def _generate_event_id() -> str:
    """Generate a unique event ID."""
    return str(uuid.uuid4())


def _timestamp_now() -> str:
    """Generate ISO format timestamp."""
    return datetime.utcnow().isoformat() + "Z"


@dataclass
class AppSessionEvent:
    """Tracks app wake/sleep for billing.

    Used to calculate total uptime and estimated costs on HF Spaces.
    """

    event_type: Literal["app_start", "heartbeat", "app_shutdown"]
    app_start_time: str  # ISO timestamp when app woke up
    uptime_sec: float  # Seconds since app_start
    analysis_count: int  # Number of analyses run this session
    analysis_time_sec: float  # Total time spent in analyses
    idle_time_sec: float  # uptime_sec - analysis_time_sec
    hourly_rate: float  # From MOSAIC_HOURLY_RATE
    idle_timeout_min: int  # From MOSAIC_IDLE_TIMEOUT_MIN
    is_hf_spaces: bool  # True if SPACE_ID env var present
    estimated_cost_usd: float  # uptime_sec / 3600 * hourly_rate
    event_id: str = field(default_factory=_generate_event_id)
    timestamp: str = field(default_factory=_timestamp_now)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class UsageEvent:
    """Tracks analysis requests.

    Logged at analysis_start and analysis_complete.
    """

    event_type: Literal["analysis_start", "analysis_complete"]
    analysis_id: str  # Links start and complete events
    session_hash: Optional[str]  # Hashed for privacy
    slide_count: int
    site_type: Optional[str] = None  # "Primary" or "Metastatic"
    cancer_subtype: Optional[str] = None  # OncoTree code or "Unknown"
    seg_config: Optional[str] = None  # "Biopsy", "Resection", or "TCGA"
    gpu_type: Optional[str] = None  # From hardware.GPU_TYPE
    duration_sec: Optional[float] = None  # On completion only
    success: Optional[bool] = None  # On completion only
    from_cache: Optional[bool] = None  # On completion only
    is_logged_in: Optional[bool] = None  # True if HF user logged in
    hf_username: Optional[str] = None  # Raw HF username (HF Spaces only)
    event_id: str = field(default_factory=_generate_event_id)
    timestamp: str = field(default_factory=_timestamp_now)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ResourceEvent:
    """Tracks compute resources per analysis.

    Logged after each slide analysis completes.
    """

    analysis_id: str  # Links to UsageEvent
    session_hash: Optional[str]  # Hashed for privacy
    total_duration_sec: float
    segmentation_duration_sec: Optional[float] = None
    ctranspath_duration_sec: Optional[float] = None
    optimus_duration_sec: Optional[float] = None
    aeon_duration_sec: Optional[float] = None
    paladin_duration_sec: Optional[float] = None
    tile_count: Optional[int] = None
    filtered_tile_count: Optional[int] = None
    batch_size: Optional[int] = None
    gpu_type: Optional[str] = None
    peak_gpu_memory_gb: Optional[float] = None
    is_logged_in: Optional[bool] = None  # True if HF user logged in
    hf_username: Optional[str] = None  # Raw HF username (HF Spaces only)
    event_id: str = field(default_factory=_generate_event_id)
    timestamp: str = field(default_factory=_timestamp_now)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class FailureEvent:
    """Tracks errors during analysis.

    Logged when an exception occurs during the analysis pipeline.
    """

    error_type: str  # Exception class name
    error_message: str  # Sanitized - no paths
    error_stage: str  # "upload", "segmentation", "ctranspath", etc.
    analysis_id: Optional[str] = None  # If available
    session_hash: Optional[str] = None  # Hashed for privacy
    slide_count: Optional[int] = None
    gpu_type: Optional[str] = None
    is_logged_in: Optional[bool] = None  # True if HF user logged in
    hf_username: Optional[str] = None  # Raw HF username (HF Spaces only)
    event_id: str = field(default_factory=_generate_event_id)
    timestamp: str = field(default_factory=_timestamp_now)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
