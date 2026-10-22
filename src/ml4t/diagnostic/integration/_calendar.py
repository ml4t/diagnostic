"""Resolve consumer overrides before authoritative producer feed metadata."""

from typing import Any

from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec


def resolve_result_calendar(
    result: Any,
    calendar: str | CalendarUnset | None = UNSET_CALENDAR,
) -> str | None:
    """Preserve omission and explicit None through the shared feed resolver."""
    config = getattr(result, "config", None)
    feed = getattr(config, "resolved_feed_spec", None)
    if feed is not None:
        return FeedSpec.from_any(feed).resolve_calendar(calendar=calendar)
    if calendar is UNSET_CALENDAR and config is not None:
        calendar = getattr(config, "resolved_calendar", UNSET_CALENDAR)
    return FeedSpec(time_contract_version=2).resolve_calendar(calendar=calendar)
