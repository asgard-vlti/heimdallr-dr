import datetime
from dataclasses import dataclass
import pathlib
import re
from typing import Optional

import pandas as pd


@dataclass
class PerformanceFile:
    path: pathlib.Path
    start_time: Optional[datetime.datetime]
    end_time: Optional[datetime.datetime]
    columns: tuple[str, ...]


class FTPerformanceBundler:
    FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
    FILE_PATTERN = re.compile(
        r"ft_(?:perf(?:ormance)?|performance)(?:_(\d{8}T\d{6}))?\.log$"
    )
    CHUNK_ROWS = 250_000
    TIMESTAMP_COLUMN = "timestamp"
    MEASUREMENT_WIDTHS = {
        "gd_snr": 6,
        "pd_snr": 6,
        "gd_bl": 6,
        "pd_tel": 4,
        "gd_tel": 4,
        "dm_piston": 4,
    }
    TRAILING_COLUMNS = ("cnt", "seq", "servo_mode")

    @classmethod
    def _base_columns(cls):
        measurements = tuple(
            f"{name}_{i}"
            for name, width in cls.MEASUREMENT_WIDTHS.items()
            for i in range(1, width + 1)
        )
        return (cls.TIMESTAMP_COLUMN, *measurements)

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
    def _read_first_data_line(fp: pathlib.Path):
        with fp.open("r") as fh:
            for ln in fh:
                s = ln.strip()
                if not s or s.startswith("#"):
                    continue
                return s
        return None

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
    def _header_columns_from_comment(cls, fp: pathlib.Path):
        base_columns = cls._base_columns()
        group_names = (cls.TIMESTAMP_COLUMN, *cls.MEASUREMENT_WIDTHS)
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

                if (
                    tuple(token.lower() for token in tokens[: len(group_names)])
                    != group_names
                ):
                    continue

                trailing_columns = []
                for token in tokens[len(group_names) :]:
                    if token.startswith("("):
                        break
                    trailing_columns.append(token)

                return (*base_columns, *trailing_columns)

        return base_columns

    @classmethod
    def _infer_columns(cls, fp: pathlib.Path):
        header_columns = cls._header_columns_from_comment(fp)
        first_data_line = cls._read_first_data_line(fp)

        if first_data_line is None:
            return header_columns

        data_count = len(first_data_line.split())
        if data_count <= len(header_columns):
            return header_columns

        base_count = len(cls._base_columns())
        extra_columns = []
        for position in range(len(header_columns) - base_count, data_count - base_count):
            if position < len(cls.TRAILING_COLUMNS):
                extra_columns.append(cls.TRAILING_COLUMNS[position])
            else:
                extra_columns.append(
                    f"aux_{position - len(cls.TRAILING_COLUMNS) + 1:02d}"
                )
        return header_columns + tuple(extra_columns)

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

    def __init__(self, path):
        """
        Parse Heimdallr performance logs under ``path``.

        The first data column is a timestamp, followed by six values for each
        baseline measurement and four for each telescope measurement. Newer
        logs also include ``cnt``, ``seq``, and ``servo_mode`` values.
        """
        p = pathlib.Path(path)
        self.path = p

        self.files = []
        if p.exists():
            for fp in p.rglob("*.log"):
                if not self.FILE_PATTERN.search(fp.name):
                    continue

                start_dt, end_dt = self._file_time_bounds(fp)
                columns = self._infer_columns(fp)
                self.files.append(
                    PerformanceFile(
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
        self.columns = self.files[0].columns if self.files else (self.TIMESTAMP_COLUMN,)

    def _iter_relevant_files(self, start_time, end_time):
        start_time = self._coerce_utc_datetime(start_time)
        end_time = self._coerce_utc_datetime(end_time)

        for sf in self.files:
            if sf.start_time is None or sf.end_time is None:
                continue
            if sf.end_time < start_time or sf.start_time > end_time:
                continue
            yield sf

    def _load_file_rows(
        self, fp: pathlib.Path, columns: tuple[str, ...], start_time, end_time
    ):
        start_time = self._coerce_utc_datetime(start_time)
        end_time = self._coerce_utc_datetime(end_time)

        start_ts = start_time.timestamp()
        end_ts = end_time.timestamp()

        reader = pd.read_csv(
            str(fp),
            comment="#",
            header=None,
            names=list(columns),
            sep=r"\s+",
            chunksize=self.CHUNK_ROWS,
            engine="python",
            na_filter=False,
            on_bad_lines="skip",
        )

        chunks = []
        for chunk in reader:
            if chunk.empty:
                continue

            time_values = pd.to_numeric(chunk[self.TIMESTAMP_COLUMN], errors="coerce")
            mask = (time_values >= start_ts) & (time_values <= end_ts)
            if not mask.any():
                continue

            chunk = chunk.loc[mask].copy()
            time_values = time_values.loc[mask]
            chunk.insert(0, "time", pd.to_datetime(time_values, unit="s", utc=True))
            chunk = chunk.drop(columns=[self.TIMESTAMP_COLUMN])
            chunk = chunk.reset_index(drop=True)
            chunks.append(chunk)

        if not chunks:
            return pd.DataFrame(columns=("time", *columns[1:]))

        df = pd.concat(chunks, ignore_index=True)
        df = df.sort_values("time").reset_index(drop=True)
        return df

    def get_performance_between(self, start_time, end_time):
        """
        Load performance rows between ``start_time`` and ``end_time``.

        Returns a tuple ``(df, frac_covered)`` where ``df`` contains the rows in
        time order and ``frac_covered`` is the number of rows divided by the
        interval duration in seconds.
        """
        start_time = self._coerce_utc_datetime(start_time)
        end_time = self._coerce_utc_datetime(end_time)

        if end_time <= start_time:
            raise ValueError("end_time must be after start_time.")

        dfs = []
        for sf in self._iter_relevant_files(start_time, end_time):
            dfs.append(self._load_file_rows(sf.path, sf.columns, start_time, end_time))

        if not dfs:
            empty = pd.DataFrame(columns=("time", *self.columns[1:]))
            return empty, 0.0

        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values("time").reset_index(drop=True)

        total_seconds = max(1.0, (end_time - start_time).total_seconds())
        frac_covered = min(1.0, float(df.shape[0]) / float(total_seconds))
        return df, frac_covered

    @classmethod
    def measurement_array(cls, df: pd.DataFrame, measurement: str):
        """Return group values aligned with ``df['time']`` as a numeric array.

        Nonnumeric values become NaN.
        """
        if measurement not in cls.MEASUREMENT_WIDTHS:
            raise ValueError(f"Unknown measurement group: {measurement}")

        columns = [
            f"{measurement}_{i}"
            for i in range(1, cls.MEASUREMENT_WIDTHS[measurement] + 1)
        ]
        missing = [column for column in columns if column not in df.columns]
        if missing and not df.empty:
            raise KeyError(f"Missing measurement columns: {', '.join(missing)}")

        values = df.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
        return values.to_numpy(dtype=float)
