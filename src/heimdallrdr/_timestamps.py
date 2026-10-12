"""UTC bounds and camera filename parsing shared by the data readers."""

import datetime
import pathlib
import re


_CAMERA_STAMP = re.compile(
    r"(?:^|_)(?:(?P<date>\d{4}-\d{2}-\d{2}))?T"
    r"(?P<hour>\d{2})(?P<separator>[:_-]?)(?P<minute>\d{2})"
    r"(?P=separator)(?P<second>\d{2})(?:\.(?P<fraction>\d+))?"
    r"(?:\.fits)?$",
    re.IGNORECASE,
)


def coerce_utc_datetime(value):
    """Interpret naive bounds as UTC and numeric bounds as Unix seconds."""
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=datetime.timezone.utc)
        return value.astimezone(datetime.timezone.utc)
    return datetime.datetime.fromtimestamp(float(value), datetime.timezone.utc)


def utc_timestamp(value):
    """Convert datetime bounds to UTC seconds without rounding numeric bounds."""
    if isinstance(value, datetime.datetime):
        return coerce_utc_datetime(value).timestamp()
    return float(value)


def _parse_camera_stamp(name, folder_day):
    match = _CAMERA_STAMP.search(pathlib.Path(name).name)
    if match is None:
        return None
    fraction = match["fraction"] or ""
    try:
        day = datetime.date.fromisoformat(match["date"]) if match["date"] else folder_day
        stamp = datetime.datetime.combine(
            day,
            datetime.time(
                int(match["hour"]),
                int(match["minute"]),
                int(match["second"]),
                int(fraction[:6].ljust(6, "0")),
                tzinfo=datetime.timezone.utc,
            ),
        )
    except ValueError:
        return None
    return stamp, fraction[:6]


def parse_camera_timestamp(name, folder_day):
    """Read a UTC camera stamp, preferring an embedded date over ``folder_day``.

    Clock fields may use colons, hyphens, underscores, or compact HHMMSS.
    Fractions longer than six digits are truncated to datetime precision.
    Bare filenames require a UTC calendar date; historical local-time or
    observing-night folder conventions must be converted before loading.
    """
    parsed = _parse_camera_stamp(name, folder_day)
    return parsed[0] if parsed is not None else None


def parse_time_seconds_and_str(name):
    """Return seconds since midnight and a colon-formatted clock for scripts."""
    parsed = _parse_camera_stamp(name, datetime.date(1900, 1, 1))
    if parsed is None:
        return None
    stamp, fraction = parsed
    seconds = (
        stamp.hour * 3600 + stamp.minute * 60 + stamp.second + stamp.microsecond / 1e6
    )
    clock = stamp.strftime("%H:%M:%S")
    if fraction:
        clock += "." + fraction
    return seconds, clock
