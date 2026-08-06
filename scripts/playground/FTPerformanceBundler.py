import os
import re
from datetime import datetime, timedelta, timezone
import pandas as pd
from pathlib import Path


class PerformanceLogExtractor:
    def __init__(self, base_path):
        """
        Initializes the extractor with the root path of the log directories.

        :param base_path: Str or Path pointing to the folder containing YYYYMMDD subdirectories.
        """
        self.base_path = Path(base_path)
        self.columns = self._generate_column_names()

    def _generate_column_names(self):
        """
        Dynamically generates the 31 columns based on the repeating pattern:
        - timestamp (1)
        - gd_snr, pd_snr, gd_bl (6 each for baselines)
        - pd_tel, gd_tel, dm_piston (4 each for telescopes)
        """
        columns = ["timestamp"]

        # 6 baselines per key
        for col in ["gd_snr", "pd_snr", "gd_bl"]:
            columns.extend([f"{col}_{i}" for i in range(1, 7)])

        # 4 telescopes per key
        for col in ["pd_tel", "gd_tel", "dm_piston"]:
            columns.extend([f"{col}_{i}" for i in range(1, 5)])

        return columns

    def _normalize_time(self, t):
        """
        Normalizes various input formats (timestamp, datetime) into standard
        UTC datetime objects and float timestamps.
        """
        if isinstance(t, (int, float)):
            dt = datetime.fromtimestamp(t, tz=timezone.utc)
            ts = float(t)
        elif isinstance(t, datetime):
            dt = t if t.tzinfo else t.replace(tzinfo=timezone.utc)
            ts = dt.timestamp()
        elif isinstance(t, str):
            dt = datetime.fromisoformat(t)
            dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            ts = dt.timestamp()
        else:
            raise TypeError(
                "Time must be a Unix timestamp (float/int) or a datetime object."
            )
        return dt, ts

    def _parse_file_datetime(self, path):
        """Extracts the start datetime from the filename and marks it as UTC."""
        match = re.search(r"ft_performance_(\d{8}T\d{6})", path.name)
        if match:
            return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(
                tzinfo=timezone.utc
            )
        return None

    def extract_rows(self, t_start, t_end):
        """
        Extracts rows between t_start and t_end as a pandas DataFrame,
        converting the timestamp column to datetime objects.

        :param t_start: Start time (Unix timestamp float or datetime object)
        :param t_end: End time (Unix timestamp float or datetime object)
        :return: pandas.DataFrame containing the filtered logs
        """
        dt_start, ts_start = self._normalize_time(t_start)
        dt_end, ts_end = self._normalize_time(t_end)

        # 1. Optimize I/O: Generate list of folders within the window
        current_date = dt_start.date() - timedelta(days=1)
        end_date = dt_end.date()

        all_files = []
        while current_date <= end_date:
            dir_name = current_date.strftime("%Y%m%d")
            dir_path = self.base_path / dir_name

            if dir_path.is_dir():
                for log_file in dir_path.glob("ft_performance_*.log"):
                    dt = self._parse_file_datetime(log_file)
                    if dt:
                        all_files.append((dt, log_file))
            current_date += timedelta(days=1)

        if not all_files:
            return pd.DataFrame(columns=self.columns)

        # 2. Sort files chronologically
        all_files.sort(key=lambda x: x[0])

        # 3. Determine which files overlap with our window
        files_to_read = []
        for i, (dt, path) in enumerate(all_files):
            next_dt = all_files[i + 1][0] if i + 1 < len(all_files) else None
            if next_dt and next_dt < dt_start:
                continue
            if dt > dt_end:
                continue
            files_to_read.append(path)

        # 4. Read and batch process the filtered files
        dataframes = []
        for path in files_to_read:
            try:
                df = pd.read_csv(
                    path, sep=r"\s+", comment="#", names=self.columns, header=None
                )
                df_filtered = df[
                    (df["timestamp"] >= ts_start) & (df["timestamp"] <= ts_end)
                ]
                if not df_filtered.empty:
                    dataframes.append(df_filtered)
            except Exception as e:
                print(f"Warning: Could not read parseable data from {path.name}: {e}")
                continue

        # 5. Combine and convert the timestamp column
        if dataframes:
            final_df = pd.concat(dataframes, ignore_index=True)

            # --- The Conversion Step ---
            # Converts the Unix floats into timezone-aware pandas datetime objects (UTC)
            final_df["timestamp"] = pd.to_datetime(
                final_df["timestamp"], unit="s", utc=True
            )

            return final_df
        else:
            return pd.DataFrame(columns=self.columns)
