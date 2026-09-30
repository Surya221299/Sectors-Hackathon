"""Unit test for the background scheduler configuration."""

from unittest.mock import patch

from app.config import settings
from app.scheduler import scheduler, start_scheduler, stop_scheduler


def test_scheduler_jobs_configured() -> None:
    # Ensure clean state
    stop_scheduler()

    with patch.object(scheduler, "start"):
        start_scheduler()

        job_ids = {job.id for job in scheduler.get_jobs()}
        assert "alert_scan" in job_ids
        assert "daily_stock_insights" in job_ids

        alert_job = scheduler.get_job("alert_scan")
        assert alert_job is not None

        # Verify alert_job trigger fields (cron at 16:00 WIB Mon-Fri)
        trigger = alert_job.trigger
        # Fields for cron trigger
        assert any(f.name == "hour" and str(f) == str(settings.alert_scan_hour) for f in trigger.fields)
        assert any(f.name == "minute" and str(f) == str(settings.alert_scan_minute) for f in trigger.fields)
        assert any(f.name == "day_of_week" and "mon-fri" in str(f).lower() for f in trigger.fields)

    stop_scheduler()
