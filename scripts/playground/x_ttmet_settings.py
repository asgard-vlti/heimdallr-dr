# %%
from dataclasses import dataclass
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import re
from tqdm.auto import tqdm


# %%
@dataclass
class SettingsFile:
    path: pathlib.Path
    start_time: datetime.datetime
    end_time: datetime.datetime
    beam: int


class BaldrTTSettingsBundler:
    @staticmethod
    def _file_datetime_from_name(fp: pathlib.Path):
        mdate = re.search(r"(\d{8}T\d{6})", fp.name)
        if mdate:
            try:
                return datetime.datetime.strptime(mdate.group(1), "%Y%m%dT%H%M%S")
            except Exception:
                return None

        m2 = re.search(r"(\d{8})", fp.name)
        if m2:
            try:
                return datetime.datetime.strptime(m2.group(1), "%Y%m%d")
            except Exception:
                return None

        return None

    @staticmethod
    def _line_to_datetime(line: str, file_dt: datetime.datetime | None):
        parts = line.split()
        if not parts:
            return None

        try:
            t = float(parts[0])
        except Exception:
            return None

        if t >= 1e7 or file_dt is None:
            try:
                return datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
            except Exception:
                return None

        try:
            return file_dt + datetime.timedelta(seconds=t)
        except Exception:
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
                    try:
                        s = raw.decode("utf-8", errors="ignore").strip()
                    except Exception:
                        continue
                    if not s or s.startswith("#"):
                        continue
                    lines.append(s)

                if len(lines) >= count:
                    return lines[-count:]

            lines = []
            for raw in buffer.splitlines():
                try:
                    s = raw.decode("utf-8", errors="ignore").strip()
                except Exception:
                    continue
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

    def __init__(self, path):
        """
        Path points to a folder of dates YYYYMMDD, which will contain settings files of the form
        btt_beam{beam}_YYYYMMDDTHHMMSS.log

        The settings files are expected to have the following format:
            # timestamp flux_threshold focus_amp focus_offset gauss_hwidth hog hol px py servo_mode ttg ttl ttxo ttyo
            1775258910.2580 10000.000 0.000 0.000 3.500 0.200 0.010 18.000 18.000 1.000 0.010 0.010 0.000 0.000

        The bundler will find all settings files under `path`, record their beam number and the
        start/end datetimes derived from the first/last timestamp lines, and expose helpers to
        load settings across files into pandas DataFrames.
        """
        p = pathlib.Path(path)
        self.path = p

        # find candidate log files recursively
        pattern = re.compile(r"btt_beam(\d+).*\.log$")
        self.files = []

        if p.exists():
            for fp in p.rglob("*.log"):
                m = pattern.search(fp.name)
                if not m:
                    continue
                beam = int(m.group(1))

                try:
                    start_dt, end_dt = self._file_time_bounds(fp)
                except Exception:
                    start_dt = None
                    end_dt = None

                sf = SettingsFile(
                    path=fp, start_time=start_dt, end_time=end_dt, beam=beam
                )
                self.files.append(sf)

        # sort files by start time
        self.files.sort(
            key=lambda x: (
                x.beam,
                x.start_time
                or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc),
            )
        )

    def get_settings_between(
        self,
        beam: int,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ):
        """
        Get settings for a beam between the specified start and end times.

        Parameters:
        - beam: The beam number to filter by.
        - start_time: The start time as a datetime object.
        - end_time: The end time as a datetime object.

        Returns:
        - settings: A pandas DataFrame containing the settings between the specified times, with columns corresponding to the settings parameters.
        - frac_covered: The fraction of the time between start_time and end_time that is covered by the settings data.
            Settings are assumed to be logged once every second or so, so this is essentially the number of settings
            entries divided by the total number of seconds between start_time and end_time.
        """
        # Normalize inputs to datetimes
        if not isinstance(start_time, datetime.datetime):
            start_time = datetime.datetime.fromtimestamp(
                float(start_time), datetime.timezone.utc
            )
        if not isinstance(end_time, datetime.datetime):
            end_time = datetime.datetime.fromtimestamp(
                float(end_time), datetime.timezone.utc
            )

        cols = [
            "flux_threshold",
            "focus_amp",
            "focus_offset",
            "gauss_hwidth",
            "hog",
            "hol",
            "px",
            "py",
            "servo_mode",
            "ttg",
            "ttl",
            "ttxo",
            "ttyo",
        ]

        data = {"time": [], **{c: [] for c in cols}}

        # helper to load a single file's rows in the requested interval
        def _load_file_rows(fp):
            file_dt = self._file_datetime_from_name(fp)

            with fp.open("r") as fh:
                for ln in fh:
                    s = ln.strip()
                    if not s or s.startswith("#"):
                        continue
                    parts = s.split()
                    if len(parts) < 14:
                        continue
                    dt = self._line_to_datetime(s, file_dt)
                    if dt is None:
                        continue

                    if dt < start_time or dt > end_time:
                        continue

                    data["time"].append(dt)
                    for i, c in enumerate(cols, start=1):
                        try:
                            data[c].append(float(parts[i]))
                        except Exception:
                            data[c].append(np.nan)

        # load rows from files whose intervals overlap request
        for sf in self.files:
            if sf.beam != beam:
                continue
            if sf.start_time is None or sf.end_time is None:
                continue
            if sf.end_time < start_time or sf.start_time > end_time:
                continue
            _load_file_rows(sf.path)

        if len(data["time"]) == 0:
            df = pd.DataFrame(data)
            return df, 0.0

        df = pd.DataFrame(data)
        try:
            df["time"] = pd.to_datetime(df["time"], utc=True)
        except Exception:
            pass

        # compute fraction covered: number of entries divided by total seconds in interval
        total_seconds = max(1.0, (end_time - start_time).total_seconds())
        frac_covered = min(1.0, float(df.shape[0]) / float(total_seconds))

        # sort by time
        df = df.sort_values("time").reset_index(drop=True)

        return df, frac_covered


# %%
base = pathlib.Path("/media/taras/T7/work/Asgard/Apr2026")


bundler = BaldrTTSettingsBundler(base)
beam = 2

print(f"Found {len(bundler.files)} settings files")

times = [
    "2026-04-04" + "02:15:40",
    "2026-04-04" + "04:46:40",
]

# convert to datetimes
times = [
    datetime.datetime.strptime(t, "%Y-%m-%d%H:%M:%S").replace(
        tzinfo=datetime.timezone.utc
    )
    for t in times
]

df, frac = bundler.get_settings_between(beam, times[0], times[1])
print(
    f"Got {df.shape[0]} settings entries for beam {beam} between {times[0]} and {times[1]}, covering {frac*100:.1f}% of the time interval."
)


servo_mode_frac = df["servo_mode"].value_counts(normalize=True)
print("Servo mode fractions:")
print(servo_mode_frac)

# %%
# servo mode 1 means loop closed, 0 means open
# find the times where the loop is closed for at least 60 seconds continuously
df["servo_closed"] = df["servo_mode"] == 1.0
servo_closed_intervals = []
current_interval_start = None

for i, row in df.iterrows():
    if row["servo_closed"]:
        if current_interval_start is None:
            current_interval_start = row["time"]
    else:
        if current_interval_start is not None:
            interval_end = row["time"]
            interval_duration = (interval_end - current_interval_start).total_seconds()
            if interval_duration >= 60.0:
                servo_closed_intervals.append((current_interval_start, interval_end))
            current_interval_start = None

# Check if we ended with an open interval
if current_interval_start is not None:
    interval_end = df["time"].iloc[-1]
    interval_duration = (interval_end - current_interval_start).total_seconds()
    if interval_duration >= 60.0:
        servo_closed_intervals.append((current_interval_start, interval_end))

print("Servo closed intervals of at least 60 seconds:")
for start, end in servo_closed_intervals:
    print(f"  {start} to {end} ({(end - start).total_seconds():.1f} seconds)")
# %%
plt.plot(df["time"], df["servo_mode"])
plt.xlim(times[0], times[1])

plt.figure()
plt.plot(df["time"], df["ttg"], label="ttg")
# %%
from heimdallrdr.DataBundler import DataBundler

db = DataBundler(base)

# %%
settings = db.load_camera_settings(times[0], times[1])
# settings is a list of tuples (time, settings_dict) where settings_dict has keys like 'fps', "readout_mode", etc.
settings_df = pd.DataFrame(
    [
        {
            "time": t,
            **s,
        }
        for t, s in settings
    ]
)
# %%
settings_df.head()
# %%
plt.plot(settings_df["time"], settings_df["fps"])
# yyaxis right
ax2 = plt.gca().twinx()
plt.plot(settings_df["time"], settings_df["camera_gain"], color="orange")

target_times = {
    "HIP 58520": [
        "2026-04-04" + "02:15:40",
        "2026-04-04" + "02:35:40",
    ],
    "HD 103862": [
        "2026-04-04" + "02:46:14",
        "2026-04-04" + "03:23:09",
    ],
    "Quasar": [
        "2026-04-04" + "03:28:18",
        "2026-04-04" + "03:35:58",
    ],
    "Bright cal": [
        "2026-04-04" + "03:43:40",
        "2026-04-04" + "04:07:06",
    ],
    "SMS_HD": [
        "2026-04-04" + "04:07:40",
        "2026-04-04" + "04:46:06",
    ],
}

for target, (tstart, tend) in target_times.items():
    tstart_dt = datetime.datetime.strptime(tstart, "%Y-%m-%d%H:%M:%S").replace(
        tzinfo=datetime.timezone.utc
    )
    tend_dt = datetime.datetime.strptime(tend, "%Y-%m-%d%H:%M:%S").replace(
        tzinfo=datetime.timezone.utc
    )
    # plt.axvspan(tstart_dt, tend_dt, alpha=0.3, label=target)

# servo_closed_intervals plotted as red vlines too
for start, end in servo_closed_intervals:
    plt.axvspan(start, end, alpha=0.09, color="red", label="servo closed")

# %%
plt.plot(df["time"], df["ttg"], label="ttg")
for start, end in servo_closed_intervals:
    plt.axvspan(start, end, alpha=0.09, color="red", label="servo closed")
# %%
from heimdallrdr.TelemBundler import load_ttmet

ttmet_fname = pathlib.Path(
    "/media/taras/T7/work/Asgard/Apr2026/20260403/btt_performance_beam1_20260403T232829.log"
)

# df_ttmet = load_ttmet(ttmet_fname, start_time=times[0], end_time=times[1])

df_ttmets = []

for start, end in servo_closed_intervals:
    df_ttmet = load_ttmet(ttmet_fname, start_time=start, end_time=end)
    print(
        f"Loaded ttmet data from {ttmet_fname} between {start} and {end}, got {df_ttmet.shape[0]} entries covering {frac*100:.1f}% of the interval."
    )

    df_ttmets.append(df_ttmet)

# %%

clip = 3.5

for i in range(len(df_ttmets)):
    tx_demeaned = df_ttmets[i]["tx"] - np.mean(df_ttmets[i]["tx"])
    ty_demeaned = df_ttmets[i]["ty"] - np.mean(df_ttmets[i]["ty"])

    clip_vals = (-clip, clip)
    ty_demeaned = np.clip(ty_demeaned, *clip_vals)
    tx_demeaned = np.clip(tx_demeaned, *clip_vals)

    plt.figure()
    plt.hist(
        tx_demeaned,
        bins=np.arange(tx_demeaned.min(), tx_demeaned.max(), 0.01),
        alpha=0.5,
        label="mx",
    )
    plt.hist(
        ty_demeaned,
        bins=np.arange(ty_demeaned.min(), ty_demeaned.max(), 0.01),
        alpha=0.5,
        label="my",
    )
    plt.legend()
    plt.xlim(-clip, clip)


# %%
import astropy.units as u

results = []

plate_scale = 13 * u.mas / u.pix
wavel = 1.22 * u.um
lam_over_D = wavel * u.rad / (8.2 * u.m)

for (start, end), df_ttmet in zip(servo_closed_intervals, df_ttmets):
    res = {}

    tx_demeaned = df_ttmet["tx"] - np.mean(df_ttmet["tx"])
    ty_demeaned = df_ttmet["ty"] - np.mean(df_ttmet["ty"])

    total_dev = np.sqrt(tx_demeaned**2 + ty_demeaned**2)
    # use 90-10 percentiles and 2.563 conversion factor
    p90 = np.percentile(total_dev, 90)
    p10 = np.percentile(total_dev, 10)
    rms_px = (p90 - p10) / 2.563
    rms_mas = (rms_px * u.pix * plate_scale).to(u.mas).value
    rms_rad = rms_mas * u.mas / lam_over_D
    rms_nm = (rms_rad * wavel / (2 * np.pi)).to(u.nm)

    res["interval_start"] = start
    res["interval_end"] = end
    res["duration_sec"] = (end - start).total_seconds()
    res["ttmet_entries"] = df_ttmet.shape[0]
    res["tx_std (px)"] = tx_demeaned.std()
    res["ty_std (px)"] = ty_demeaned.std()
    res["rms tip/tilt (mas)"] = rms_mas
    res["rms tip/tilt (nm)"] = rms_nm.value

    results.append(res)

results_df = pd.DataFrame(results)

# %%
beams = [1, 2, 3, 4]
# now df over beams in columns (just rms nm)

results = []  # each item is still an observation
for (start, end), df_ttmet in zip(servo_closed_intervals, df_ttmets):
    res = {}
    res = {"interval_start": start, "interval_end": end}
    res["duration_sec"] = (end - start).total_seconds()

    for beam in beams:
        ttmet_fname = pathlib.Path(
            f"/media/taras/T7/work/Asgard/Apr2026/20260403/btt_performance_beam{beam}_20260403T232829.log"
        )
        df_ttmet = load_ttmet(ttmet_fname, start_time=start, end_time=end)
        tx_demeaned = df_ttmet["tx"] - np.mean(df_ttmet["tx"])
        ty_demeaned = df_ttmet["ty"] - np.mean(df_ttmet["ty"])
        total_dev = np.sqrt(tx_demeaned**2 + ty_demeaned**2)
        p90 = np.percentile(total_dev, 90)
        p10 = np.percentile(total_dev, 10)
        rms_px = (p90 - p10) / 2.563
        rms_mas = (rms_px * u.pix * plate_scale).to(u.mas).value
        rms_rad = rms_mas * u.mas / lam_over_D
        rms_nm = (rms_rad * wavel / (2 * np.pi)).to(u.nm)

        res[f"beam{beam}_ttmet_entries"] = df_ttmet.shape[0]
        res[f"beam{beam}_rms_nm"] = rms_nm.value
    results.append(res)

results_df = pd.DataFrame(results)

# save to csv
night_name = "20260403"
results_df.to_csv(f"btt_ttmet_summary_{night_name}.csv", index=False)
