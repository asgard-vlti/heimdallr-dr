# heimdallr-dr

Reader time bounds use UTC: naive datetimes are interpreted as UTC, aware
datetimes are converted to UTC, and numeric bounds are Unix seconds. Camera
frame times retain the existing timezone-free `datetime64[us]` representation
of UTC; telemetry times retain their UTC timezone.

Camera readers accept clocks separated by `:`, `-`, or `_`, or compact `HHMMSS`,
with optional fractions. They accept both `baldr1_T12-34-56.123.fits` and dated
names such as `baldr1_2026-10-10T12-34-56.123.fits`; an embedded date overrides
the directory date. Fractions beyond six digits are truncated to microseconds.
Bare names require UTC calendar-date folders. Historical local timestamps or
observing-night folder dates must be converted before loading.
