import datetime
from dataclasses import dataclass
import pathlib
import re
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class FringeSettingsFile:
    path: pathlib.Path
    start_time: Optional[datetime.datetime]
    end_time: Optional[datetime.datetime]
    columns: tuple[str, ...]


class FTSettingsBundler:
    FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
    DEFAULT_COLUMNS = (
        "timestamp",
        "n_gd_boxcar",
        "gd_threshold",
        "pd_threshold",
        "gd_search_reset",
        "offload_time_ms",
        "offload_gd_gain",
        "gd_gain",
        "kp",
    )

    @staticmethod
    def _file_datetime_from_name(fp: pathlib.Path):
        mdate = re.search(r"(\d{8}T\d{6})", fp.name)
        if mdate:
            return datetime.datetime.strptime(mdate.group(1), "%Y%m%dT%H%M%S").replace(
                tzinfo=datetime.timezone.utc
            )

        m2 = re.search(r"(\d{8})", fp.name)
        if m2:
            return datetime.datetime.strptime(m2.group(1), "%Y%m%d").replace(
                tzinfo=datetime.timezone.utc
            )

        return None

    @classmethod
    def _parse_float_token(cls, token: str):
        if not cls.FLOAT_RE.match(token):
            return None
        return float(token)

    @staticmethod
    def _coerce_utc_datetime(value):
        if isinstance(value, datetime.datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=datetime.timezone.utc)
            return value.astimezone(datetime.timezone.utc)

        return datetime.datetime.fromtimestamp(float(value), datetime.timezone.utc)

    @classmethod
    def _line_to_datetime(cls, line: str, file_dt: Optional[datetime.datetime]):
        parts = line.split()
        if not parts:
            return None

        t = cls._parse_float_token(parts[0])
        if t is None:
            return None

        if t >= 1e7 or file_dt is None:
            return datetime.datetime.fromtimestamp(t, datetime.timezone.utc)

        return file_dt + datetime.timedelta(seconds=t)

    @staticmethod
    def _read_first_data_lines(fp: pathlib.Path, count: int = 2):
        lines = []
        with fp.open("r") as fh:
            for ln in fh:
                s = ln.strip()
                if not s or s.startswith("#"):
                    continue
                lines.append(s)
                if len(lines) >= count:
                    break
        return lines

    @staticmethod
    def _read_last_data_lines(fp: pathlib.Path, count: int = 2, chunk_size: int = 8192):
        with fp.open("rb") as fh:
            fh.seek(0, 2)
            position = fh.tell()
            buffer = b""

            while position > 0:
                read_size = min(chunk_size, position)
                position -= read_size
                fh.seek(position)
                buffer = fh.read(read_size) + buffer

                lines = []
                for raw in buffer.splitlines():
                    s = raw.decode("utf-8", errors="ignore").strip()
                    if not s or s.startswith("#"):
                        continue
                    lines.append(s)

                if len(lines) >= count:
                    return lines[-count:]

            lines = []
            for raw in buffer.splitlines():
                s = raw.decode("utf-8", errors="ignore").strip()
                if not s or s.startswith("#"):
                    continue
                lines.append(s)

            return lines[-count:]

    @classmethod
    def _file_time_bounds(cls, fp: pathlib.Path):
        file_dt = cls._file_datetime_from_name(fp)
        first_lines = cls._read_first_data_lines(fp, count=2)
        last_lines = cls._read_last_data_lines(fp, count=2)

        start_dt = None
        for line in first_lines:
            start_dt = cls._line_to_datetime(line, file_dt)
            if start_dt is not None:
                break

        end_dt = None
        for line in reversed(last_lines):
            end_dt = cls._line_to_datetime(line, file_dt)
            if end_dt is not None:
                break

        return start_dt, end_dt

    @classmethod
    def _file_columns_from_header(cls, fp: pathlib.Path):
        with fp.open("r") as fh:
            for ln in fh:
                s = ln.strip()
                if not s:
                    continue
                if not s.startswith("#"):
                    break

                header = s.lstrip("#").strip()
                if not header:
                    continue

                tokens = header.split()
                if not tokens:
                    continue

                if tokens[0].lower() not in {"timestamp", "time"}:
                    continue

                columns = []
                for token in tokens:
                    if token.startswith("("):
                        break
                    columns.append(token)

                if columns:
                    return tuple(columns)

        return cls.DEFAULT_COLUMNS

    def __init__(self, path):
        """
        Parse Heimdallr fringe-tracker settings logs under ``path``.

        The expected layout is ``base/YYYYMMDD/ft_settings_YYYYMMDDTHHMMSS.log``.
        Each file is treated as a single settings stream with a header comment
        describing the columns and timestamp values in the first column.
        """
        p = pathlib.Path(path)
        self.path = p

        pattern = re.compile(r"ft_settings_(\d{8}T\d{6})\.log$")
        self.files = []

        if p.exists():
            for fp in p.rglob("*.log"):
                if not pattern.search(fp.name):
                    continue

                start_dt, end_dt = self._file_time_bounds(fp)
                columns = self._file_columns_from_header(fp)

                self.files.append(
                    FringeSettingsFile(
                        path=fp,
                        start_time=start_dt,
                        end_time=end_dt,
                        columns=columns,
                    )
                )

        self.files.sort(
            key=lambda x: x.start_time
            or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
        )
        self.columns = self.files[0].columns if self.files else self.DEFAULT_COLUMNS

    def get_settings_between(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ):
        """
        Get fringe settings between the specified times.

        Returns a tuple ``(settings_df, frac_covered)`` where ``settings_df`` is
        sorted by time and ``frac_covered`` is the number of samples divided by
        the interval duration in seconds.
        """
        start_time = self._coerce_utc_datetime(start_time)
        end_time = self._coerce_utc_datetime(end_time)

        rows = []

        for sf in self.files:
            if sf.start_time is None or sf.end_time is None:
                continue
            if sf.end_time < start_time or sf.start_time > end_time:
                continue

            file_dt = self._file_datetime_from_name(sf.path)

            with sf.path.open("r") as fh:
                for ln in fh:
                    s = ln.strip()
                    if not s or s.startswith("#"):
                        continue

                    parts = s.split()
                    dt = self._line_to_datetime(s, file_dt)
                    if dt is None or dt < start_time or dt > end_time:
                        continue

                    row = {"time": dt}
                    for i, col in enumerate(sf.columns[1:], start=1):
                        if i >= len(parts):
                            row[col] = np.nan
                            continue

                        value = self._parse_float_token(parts[i])
                        row[col] = value if value is not None else np.nan

                    rows.append(row)

        if not rows:
            return pd.DataFrame(columns=("time", *self.columns[1:])), 0.0

        df = pd.DataFrame(rows)
        df = df.sort_values("time").reset_index(drop=True)
        df["time"] = pd.to_datetime(df["time"], utc=True)

        total_seconds = max(1.0, (end_time - start_time).total_seconds())
        frac_covered = min(1.0, float(df.shape[0]) / float(total_seconds))

        return df, frac_covered
