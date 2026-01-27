"""Unit tests for datetime utilities."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from eval_hub.utils.datetime_utils import (
    ensure_timezone_aware,
    parse_iso_datetime,
    safe_duration_seconds,
    utcnow,
)


class TestDatetimeUtils:
    """Test datetime utility functions."""

    def test_utcnow_returns_timezone_aware(self):
        """Test utcnow returns timezone-aware datetime in UTC."""
        result = utcnow()

        assert isinstance(result, datetime)
        assert result.tzinfo is not None
        assert result.tzinfo == UTC

    def test_parse_iso_datetime_with_z_suffix(self):
        """Test parsing ISO datetime with 'Z' suffix."""
        result = parse_iso_datetime("2024-01-15T10:30:00Z")

        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30
        assert result.second == 0
        assert result.tzinfo == UTC

    def test_parse_iso_datetime_with_timezone_offset(self):
        """Test parsing ISO datetime with timezone offset."""
        result = parse_iso_datetime("2024-01-15T10:30:00+05:00")

        expected_offset = timezone(timedelta(hours=5))
        assert result.year == 2024
        assert result.tzinfo == expected_offset

    def test_parse_iso_datetime_without_timezone(self):
        """Test parsing ISO datetime without timezone returns naive datetime."""
        result = parse_iso_datetime("2024-01-15T10:30:00")

        assert result.year == 2024
        # Note: fromisoformat returns naive datetime when no tz in string
        assert result.tzinfo is None

    def test_parse_iso_datetime_empty_string_raises(self):
        """Test parsing empty string raises ValueError."""
        with pytest.raises(ValueError, match="Empty datetime string"):
            parse_iso_datetime("")

    def test_parse_iso_datetime_invalid_format_raises(self):
        """Test parsing invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Unable to parse datetime string"):
            parse_iso_datetime("not-a-datetime")

    def test_ensure_timezone_aware_with_naive_datetime(self):
        """Test ensure_timezone_aware adds UTC to naive datetime."""
        naive_dt = datetime(2024, 1, 15, 10, 30, 0)
        assert naive_dt.tzinfo is None

        result = ensure_timezone_aware(naive_dt)

        assert result.tzinfo == UTC  # Note: key point of this test
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30

    def test_ensure_timezone_aware_with_aware_datetime(self):
        """Test ensure_timezone_aware preserves existing timezone."""
        aware_dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone(timedelta(hours=5)))

        result = ensure_timezone_aware(aware_dt)

        assert result is aware_dt
        assert result.tzinfo == timezone(timedelta(hours=5))

    def test_safe_duration_seconds_with_naive_datetimes(self):
        """Test safe_duration_seconds with naive datetimes."""
        start = datetime(2024, 1, 15, 10, 0, 0)
        end = datetime(2024, 1, 15, 10, 5, 30)

        result = safe_duration_seconds(end, start)

        assert result == 330.0  # 5 minutes and 30 seconds

    def test_safe_duration_seconds_with_aware_datetimes(self):
        """Test safe_duration_seconds with timezone-aware datetimes."""
        start = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
        end = datetime(2024, 1, 15, 10, 5, 30, tzinfo=UTC)

        result = safe_duration_seconds(end, start)

        assert result == 330.0  # 5 minutes and 30 seconds

    def test_safe_duration_seconds_with_mixed_datetimes(self):
        """Test safe_duration_seconds with mixed naive and aware datetimes."""
        start = datetime(2024, 1, 15, 10, 0, 0)  # naive
        end = datetime(2024, 1, 15, 10, 5, 30, tzinfo=UTC)  # aware

        result = safe_duration_seconds(end, start)

        assert result == 330.0  # 5 minutes and 30 seconds
