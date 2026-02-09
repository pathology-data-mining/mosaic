"""Main telemetry tracker singleton for Mosaic.

This module provides the TelemetryTracker singleton that:
- Manages app session lifecycle (start, heartbeat, shutdown)
- Logs usage, resource, and failure events
- Handles HuggingFace Dataset uploads
"""

import threading
from datetime import datetime
from typing import Optional

from loguru import logger

from mosaic.telemetry.config import TelemetryConfig
from mosaic.telemetry.events import (
    AppSessionEvent,
    UsageEvent,
    ResourceEvent,
    FailureEvent,
)
from mosaic.telemetry.storage import TelemetryStorage
from mosaic.telemetry.utils import hash_session_id, sanitize_error_message


class TelemetryTracker:
    """Singleton tracker for Mosaic telemetry.

    Manages telemetry collection for the Gradio web app including:
    - App session lifecycle (start, heartbeat, shutdown)
    - Analysis usage events
    - Resource consumption metrics
    - Failure tracking

    Usage:
        tracker = TelemetryTracker.get_instance()
        tracker.log_app_start()
        tracker.log_usage_event(event_type="analysis_start", ...)
    """

    _instance: Optional["TelemetryTracker"] = None
    _lock = threading.Lock()

    def __init__(self, config: Optional[TelemetryConfig] = None):
        """Initialize tracker.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or TelemetryConfig()
        self.storage = TelemetryStorage(self.config.base_dir)

        # Session state
        self._app_start_time: Optional[datetime] = None
        self._analysis_count: int = 0
        self._analysis_time_sec: float = 0.0
        self._session_lock = threading.Lock()

    @classmethod
    def get_instance(
        cls, config: Optional[TelemetryConfig] = None
    ) -> "TelemetryTracker":
        """Get or create the singleton tracker instance.

        Args:
            config: Optional configuration (only used on first call)

        Returns:
            The singleton TelemetryTracker instance
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance. Primarily for testing."""
        with cls._lock:
            cls._instance = None

    def _is_enabled(self) -> bool:
        """Check if telemetry is enabled."""
        return self.config.enabled

    def _get_session_metrics(self) -> dict:
        """Get current session metrics.

        Returns:
            Dict with uptime_sec, analysis_count, analysis_time_sec, idle_time_sec
        """
        with self._session_lock:
            if self._app_start_time is None:
                return {
                    "uptime_sec": 0.0,
                    "analysis_count": 0,
                    "analysis_time_sec": 0.0,
                    "idle_time_sec": 0.0,
                }

            uptime_sec = (datetime.utcnow() - self._app_start_time).total_seconds()
            idle_time_sec = uptime_sec - self._analysis_time_sec

            return {
                "uptime_sec": uptime_sec,
                "analysis_count": self._analysis_count,
                "analysis_time_sec": self._analysis_time_sec,
                "idle_time_sec": max(0, idle_time_sec),
            }

    def _calculate_cost(self, uptime_sec: float) -> float:
        """Calculate estimated cost based on uptime.

        Args:
            uptime_sec: Total uptime in seconds

        Returns:
            Estimated cost in USD
        """
        return (uptime_sec / 3600) * self.config.hourly_rate

    # =========================================================================
    # App Session Events
    # =========================================================================

    def log_app_start(self) -> None:
        """Log app start event. Called when Gradio app launches."""
        if not self._is_enabled():
            return

        with self._session_lock:
            self._app_start_time = datetime.utcnow()
            self._analysis_count = 0
            self._analysis_time_sec = 0.0

        event = AppSessionEvent(
            event_type="app_start",
            app_start_time=self._app_start_time.isoformat() + "Z",
            uptime_sec=0.0,
            analysis_count=0,
            analysis_time_sec=0.0,
            idle_time_sec=0.0,
            hourly_rate=self.config.hourly_rate,
            idle_timeout_min=self.config.idle_timeout_min,
            is_hf_spaces=self.config.is_hf_spaces,
            estimated_cost_usd=0.0,
        )
        self.storage.write_session_event(event)
        logger.info(f"Telemetry: App started, logging to {self.config.base_dir}")

    def log_heartbeat(self) -> None:
        """Log heartbeat event. Called periodically to track uptime."""
        if not self._is_enabled():
            return

        if self._app_start_time is None:
            return

        metrics = self._get_session_metrics()
        event = AppSessionEvent(
            event_type="heartbeat",
            app_start_time=self._app_start_time.isoformat() + "Z",
            uptime_sec=metrics["uptime_sec"],
            analysis_count=metrics["analysis_count"],
            analysis_time_sec=metrics["analysis_time_sec"],
            idle_time_sec=metrics["idle_time_sec"],
            hourly_rate=self.config.hourly_rate,
            idle_timeout_min=self.config.idle_timeout_min,
            is_hf_spaces=self.config.is_hf_spaces,
            estimated_cost_usd=self._calculate_cost(metrics["uptime_sec"]),
        )
        self.storage.write_session_event(event)

    def log_app_shutdown(self) -> None:
        """Log app shutdown event. Called when app is terminating."""
        if not self._is_enabled():
            return

        if self._app_start_time is None:
            return

        metrics = self._get_session_metrics()
        event = AppSessionEvent(
            event_type="app_shutdown",
            app_start_time=self._app_start_time.isoformat() + "Z",
            uptime_sec=metrics["uptime_sec"],
            analysis_count=metrics["analysis_count"],
            analysis_time_sec=metrics["analysis_time_sec"],
            idle_time_sec=metrics["idle_time_sec"],
            hourly_rate=self.config.hourly_rate,
            idle_timeout_min=self.config.idle_timeout_min,
            is_hf_spaces=self.config.is_hf_spaces,
            estimated_cost_usd=self._calculate_cost(metrics["uptime_sec"]),
        )
        self.storage.write_session_event(event)
        logger.info(
            f"Telemetry: App shutdown after {metrics['uptime_sec']:.1f}s, "
            f"{metrics['analysis_count']} analyses, "
            f"estimated cost: ${self._calculate_cost(metrics['uptime_sec']):.2f}"
        )

    # =========================================================================
    # Usage Events
    # =========================================================================

    def log_usage_event(
        self,
        event_type: str,
        analysis_id: str,
        slide_count: int,
        session_hash: Optional[str] = None,
        site_type: Optional[str] = None,
        cancer_subtype: Optional[str] = None,
        seg_config: Optional[str] = None,
        gpu_type: Optional[str] = None,
        duration_sec: Optional[float] = None,
        success: Optional[bool] = None,
        from_cache: Optional[bool] = None,
        is_logged_in: Optional[bool] = None,
        hf_username: Optional[str] = None,
    ) -> None:
        """Log a usage event (analysis_start or analysis_complete).

        Args:
            event_type: "analysis_start" or "analysis_complete"
            analysis_id: Unique ID linking start and complete events
            slide_count: Number of slides in the analysis
            session_hash: Gradio session hash (will be hashed for privacy)
            site_type: "Primary" or "Metastatic"
            cancer_subtype: OncoTree code or "Unknown"
            seg_config: "Biopsy", "Resection", or "TCGA"
            gpu_type: GPU type string
            duration_sec: Analysis duration (for analysis_complete only)
            success: Whether analysis succeeded (for analysis_complete only)
            from_cache: True if result was served from cache (for analysis_complete only)
            is_logged_in: True if HF user logged in
            hf_username: HF username (HF Spaces only)
        """
        if not self._is_enabled():
            return

        event = UsageEvent(
            event_type=event_type,
            analysis_id=analysis_id,
            session_hash=hash_session_id(session_hash),
            slide_count=slide_count,
            site_type=site_type,
            cancer_subtype=cancer_subtype,
            seg_config=seg_config,
            gpu_type=gpu_type,
            duration_sec=duration_sec,
            success=success,
            from_cache=from_cache,
            is_logged_in=is_logged_in,
            hf_username=hf_username,
        )
        self.storage.write_usage_event(event)

        # Update session metrics on analysis complete
        if event_type == "analysis_complete" and duration_sec is not None:
            with self._session_lock:
                self._analysis_count += 1
                # Don't count duration if result was served from cache
                if not from_cache:
                    self._analysis_time_sec += duration_sec

    # =========================================================================
    # Resource Events
    # =========================================================================

    def log_resource_event(
        self,
        analysis_id: str,
        total_duration_sec: float,
        session_hash: Optional[str] = None,
        segmentation_duration_sec: Optional[float] = None,
        ctranspath_duration_sec: Optional[float] = None,
        optimus_duration_sec: Optional[float] = None,
        aeon_duration_sec: Optional[float] = None,
        paladin_duration_sec: Optional[float] = None,
        tile_count: Optional[int] = None,
        filtered_tile_count: Optional[int] = None,
        batch_size: Optional[int] = None,
        gpu_type: Optional[str] = None,
        peak_gpu_memory_gb: Optional[float] = None,
        is_logged_in: Optional[bool] = None,
        hf_username: Optional[str] = None,
    ) -> None:
        """Log a resource event for a completed analysis.

        Args:
            analysis_id: Links to the corresponding UsageEvent
            total_duration_sec: Total slide processing time
            session_hash: Gradio session hash (will be hashed for privacy)
            segmentation_duration_sec: Time for tissue segmentation
            ctranspath_duration_sec: Time for CTransPath feature extraction
            optimus_duration_sec: Time for Optimus feature extraction
            aeon_duration_sec: Time for Aeon inference
            paladin_duration_sec: Time for Paladin inference
            tile_count: Number of tiles before filtering
            filtered_tile_count: Number of tiles after filtering
            batch_size: Batch size used
            gpu_type: GPU type string
            peak_gpu_memory_gb: Peak GPU memory usage in GB
            is_logged_in: True if HF user logged in
            hf_username: HF username (HF Spaces only)
        """
        if not self._is_enabled():
            return

        event = ResourceEvent(
            analysis_id=analysis_id,
            session_hash=hash_session_id(session_hash),
            total_duration_sec=total_duration_sec,
            segmentation_duration_sec=segmentation_duration_sec,
            ctranspath_duration_sec=ctranspath_duration_sec,
            optimus_duration_sec=optimus_duration_sec,
            aeon_duration_sec=aeon_duration_sec,
            paladin_duration_sec=paladin_duration_sec,
            tile_count=tile_count,
            filtered_tile_count=filtered_tile_count,
            batch_size=batch_size,
            gpu_type=gpu_type,
            peak_gpu_memory_gb=peak_gpu_memory_gb,
            is_logged_in=is_logged_in,
            hf_username=hf_username,
        )
        self.storage.write_resource_event(event)

    # =========================================================================
    # Failure Events
    # =========================================================================

    def log_failure_event(
        self,
        error_type: str,
        error_message: str,
        error_stage: str,
        analysis_id: Optional[str] = None,
        session_hash: Optional[str] = None,
        slide_count: Optional[int] = None,
        gpu_type: Optional[str] = None,
        is_logged_in: Optional[bool] = None,
        hf_username: Optional[str] = None,
    ) -> None:
        """Log a failure event.

        Args:
            error_type: Exception class name (e.g., "ValueError")
            error_message: Error message (will be sanitized)
            error_stage: Pipeline stage where error occurred
            analysis_id: Links to UsageEvent if available
            session_hash: Gradio session hash (will be hashed for privacy)
            slide_count: Number of slides if known
            gpu_type: GPU type string
            is_logged_in: True if HF user logged in
            hf_username: HF username (HF Spaces only)
        """
        if not self._is_enabled():
            return

        event = FailureEvent(
            error_type=error_type,
            error_message=sanitize_error_message(error_message),
            error_stage=error_stage,
            analysis_id=analysis_id,
            session_hash=hash_session_id(session_hash),
            slide_count=slide_count,
            gpu_type=gpu_type,
            is_logged_in=is_logged_in,
            hf_username=hf_username,
        )
        self.storage.write_failure_event(event)

    # =========================================================================
    # Upload
    # =========================================================================

    def upload_to_hf(self) -> bool:
        """Upload telemetry to HuggingFace Dataset if configured.

        Returns:
            True if upload succeeded, False otherwise
        """
        if not self.config.hf_repo:
            logger.debug("No HF repo configured, skipping telemetry upload")
            return False

        return self.storage.upload_to_hf_dataset(self.config.hf_repo)
