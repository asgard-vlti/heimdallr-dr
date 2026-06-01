# %%
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import re
from tqdm.auto import tqdm


def load_ttsettings(fname):
    """
    Loads a ttsettings file into a pd dataframe

    fname: btt_beam1_settings_20260403T021244.log
    # timestamp flux_threshold focus_amp focus_offset gauss_hwidth hog hol px py servo_mode ttg ttl ttxo ttyo
    1775182365.2593 10000.000 0.000 0.000 3.500 0.200 0.010 16.000 18.000 0.000 0.010 0.010 0.000 0.000
    1775182366.2654 10000.000 0.000 0.000 3.500 0.200 0.010 16.000 18.000 0.000 0.010 0.010 0.000 0.000
    1775182367.2728 10000.000 0.000 0.000 3.500 0.200 0.010 16.000 18.000 0.000 0.010 0.010 0.000 0.000
    1775182368.2792 10000.000 0.000 0.000 3.500 0.200 0.010 16.000 18.000 0.000 0.010 0.010 0.000 0.000
    1775182369.2829 10000.000 0.000 0.000 3.500 0.200 0.010 16.000 18.000 0.000 0.010 0.010 0.000 0.000
    1775182370.2896 10000.000 0.000 0.000 3.500 0.200 0.010 16.000 18.000 0.000 0.010 0.010 0.000 0.000
    1775182371.3053 10000.000 0.000 0.000 3.500 0.200 0.010 16.000 18.000 0.000 0.010 0.010 0.000 0.000

    timestamps converted to datetime objects, using the fname to get the date too
    """
    p = pathlib.Path(fname)

    # Attempt to extract a date token from the filename (e.g. 20260403T021244)
    m = re.search(r"(\d{8}T\d{6})", p.name)
    if m:
        try:
            file_dt = datetime.datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
        except Exception:
            file_dt = None
    else:
        # fallback to plain YYYYMMDD
        m2 = re.search(r"(\d{8})", p.name)
        if m2:
            try:
                file_dt = datetime.datetime.strptime(m2.group(1), "%Y%m%d")
            except Exception:
                file_dt = None
        else:
            file_dt = None

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

    with p.open("r") as fh:
        for ln in fh:
            s = ln.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            parts = s.split()
            # timestamp + 13 fields => total 14 tokens
            if len(parts) < 14:
                continue
            try:
                t = float(parts[0])
            except Exception:
                continue

            # convert timestamp to datetime
            # If value looks like a full epoch (>= 1e7), use utcfromtimestamp.
            # Otherwise, if filename contained a date, treat t as seconds offset from that date.
            if t >= 1e7 or file_dt is None:
                try:
                    dt = datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
                except Exception:
                    dt = None
            else:
                try:
                    # file_dt may include time; if so, use that as base
                    base = file_dt
                    dt = base + datetime.timedelta(seconds=t)
                except Exception:
                    dt = None

            data["time"].append(dt)

            # parse remaining 13 numeric settings
            for i, c in enumerate(cols, start=1):
                try:
                    data[c].append(float(parts[i]))
                except Exception:
                    data[c].append(np.nan)

    # Build DataFrame and convert time column to pandas datetime if possible
    df = pd.DataFrame(data)
    try:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    except Exception:
        pass

    return df


def find_longest_run(bool_arr):
    """
    Given a boolean array, find the longest consecutive run of True values and return its length and start/end indices.
    """
    values = np.asarray(bool_arr)

    if len(values) == 0:
        return 0, 0, 0

    max_length = 0
    max_start = 0
    max_end = 0

    current_length = 0
    current_start = 0

    for i, val in enumerate(values):
        if val:
            if current_length == 0:
                current_start = i
            current_length += 1
        else:
            if current_length > max_length:
                max_length = current_length
                max_start = current_start
                max_end = i - 1
            current_length = 0

    # Check one more time in case the array ends with True
    if current_length > max_length:
        max_length = current_length
        max_start = current_start
        max_end = len(bool_arr) - 1

    return max_length, max_start, max_end


def load_ttmet(fname, start_time=None, end_time=None):
    """
    Loads a ttmet file into a pd dataframe

    time tx ty mx my
    1775184464.9896 0.126 -0.157 0.064 -0.025
    prev 0.202 -0.146 0.066 -0.027
    prev 0.307 -0.120 0.068 -0.027
    prev 0.304 -0.272 0.071 -0.030
    1775184464.9944 0.478 -0.337 0.075 -0.033
    prev 0.518 -0.337 0.079 -0.036
    prev 0.548 -0.186 0.084 -0.037
    prev 0.468 0.256 0.088 -0.035
    prev 0.523 0.458 0.092 -0.030
    1775184464.9994 0.441 0.562 0.095 -0.024
    prev 0.219 0.657 0.097 -0.017
    prev 0.105 0.376 0.097 -0.013
    prev -0.033 0.132 0.095 -0.012
    prev -0.219 0.031 0.092 -0.011
    1775184465.0047 -0.217 0.092 0.089 -0.010

    ...
    in cases of "prev" in time, use the next timestamp and interpolate linearly

    Optional start_time/end_time bounds can be supplied to only materialize rows
    in the requested interval. When bounds are provided, the parser streams the
    file and avoids storing rows outside the window.
    """

    # Accept either a path-like or a string
    p = pathlib.Path(fname)

    # convert start_time and end_time to seconds to compare directly with the timestamps in the file
    if start_time is not None:
        if isinstance(start_time, datetime.datetime):
            start_time_sec = start_time.timestamp()
        else:
            start_time_sec = float(start_time)
    else:
        start_time_sec = None
    if end_time is not None:
        if isinstance(end_time, datetime.datetime):
            end_time_sec = end_time.timestamp()
        else:
            end_time_sec = float(end_time)
    else:
        end_time_sec = None

    data = {"time": [], "tx": [], "ty": [], "mx": [], "my": []}

    # Count lines first so tqdm can display a real progress bar with completion.
    with p.open("r") as fh:
        total_lines = sum(1 for _ in fh)

    with p.open("r") as fh:
        prev_time = None
        # to hold rows with "prev" time until we get a real timestamp to assign them
        # (then interpolate linearly if there are multiple "prev" rows in a row)
        prev_stack = []

        for ln in tqdm(fh, total=total_lines, desc="load_ttmet", unit="line"):
            s = ln.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 5:
                continue

            time_str, tx_str, ty_str, mx_str, my_str = parts[:5]

            if time_str == "prev":
                if prev_time is None:
                    continue  # skip "prev" rows until we get a real timestamp
                else:
                    prev_stack.append((tx_str, ty_str, mx_str, my_str))
            else:
                try:
                    t = float(time_str)
                except Exception:
                    continue

                # Now add the current real timestamp row
                if (start_time_sec is not None and t < start_time_sec) or (
                    end_time_sec is not None and t > end_time_sec
                ):
                    continue  # skip rows outside the requested time window

                # If we have "prev" rows stacked, assign them timestamps by interpolating between prev_time and t
                if prev_stack:
                    n_prev = len(prev_stack)
                    for i, (tx_s, ty_s, mx_s, my_s) in enumerate(prev_stack):
                        interp_time = prev_time + (t - prev_time) * (i + 1) / (
                            n_prev + 1
                        )
                        data["time"].append(
                            datetime.datetime.fromtimestamp(
                                interp_time, datetime.timezone.utc
                            )
                        )
                        data["tx"].append(float(tx_s))
                        data["ty"].append(float(ty_s))
                        data["mx"].append(float(mx_s))
                        data["my"].append(float(my_s))
                    prev_stack = []

                data["time"].append(
                    datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
                )
                try:
                    data["tx"].append(float(tx_str))
                    data["ty"].append(float(ty_str))
                    data["mx"].append(float(mx_str))
                    data["my"].append(float(my_str))
                except Exception:
                    data["tx"].append(np.nan)
                    data["ty"].append(np.nan)
                    data["mx"].append(np.nan)
                    data["my"].append(np.nan)

                prev_time = t

    df = pd.DataFrame(data)
    return df

# base = pathlib.Path("/media/taras/T7/work/Asgard/Apr2026")

# ttmet_fname = "btt_performance_minimal.log"
# pth = pathlib.Path(__file__).parent / "data"

# ttmet_minimal = load_ttmet(pth / ttmet_fname)
# print(ttmet_minimal.head())

# %%
# from log manually for now
# target name -> [start time, end time]
target_times = {
    # "HD51914": [
    #     datetime.datetime(2026, 4, 3, 23, 29, 51, tzinfo=datetime.timezone.utc),
    #     datetime.datetime(2026, 4, 3, 23, 53, 2, tzinfo=datetime.timezone.utc),
    # ],
    # "HIP58520": [
    #     datetime.datetime(2026, 4, 4, 0, 21, 9, tzinfo=datetime.timezone.utc),
    #     datetime.datetime(2026, 4, 4, 0, 57, 12, tzinfo=datetime.timezone.utc),
    # ],
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
        "2026-04-04" + "03:40:58",
    ],
    "SMS_HD": [
        "2026-04-04" + "03:43:40",
        "2026-04-04" + "04:46:06",
    ],
}

for target, (start_str, end_str) in target_times.items():
    if isinstance(start_str, str):
        start_str = datetime.datetime.strptime(start_str, "%Y-%m-%d%H:%M:%S").replace(
            tzinfo=datetime.timezone.utc
        )
    if isinstance(end_str, str):
        end_str = datetime.datetime.strptime(end_str, "%Y-%m-%d%H:%M:%S").replace(
            tzinfo=datetime.timezone.utc
        )
    target_times[target] = (start_str, end_str)


target = "SMS_HD"
# %%
ttmet_fname = "btt_performance_beam1_20260403T232829.log"
pth = pathlib.Path(__file__).parent / "data"

ttmet = load_ttmet(
    pth / ttmet_fname,
    start_time=target_times[target][0],
    end_time=target_times[target][1],
)
# %%
settings_fname = "btt_beam1_settings_20260403T232829.log"

settings = load_ttsettings(pth / settings_fname)
# %%
print(f"Loaded all data")
print(f"Settings times: {settings['time'].min()} to {settings['time'].max()}")
print(f"TTMET times: {ttmet['time'].min()} to {ttmet['time'].max()}")

# %%

rows = []
for itarget, (start_str, end_str) in target_times.items():

    start = start_str
    end = end_str

    duration = (end - start).total_seconds() / 60
    n_ttmet = ttmet[(ttmet["time"] >= start) & (ttmet["time"] <= end)].shape[0]
    rows.append(
        {
            "Target": itarget,
            "Start Time (UTC)": start,
            "End Time (UTC)": end,
            "Duration (min)": f"{duration:.1f}",
            "N_TTMET": n_ttmet,
            "frac servo_mode 1": f"{(settings[(settings['time'] >= start) & (settings['time'] <= end)]['servo_mode'] == 1).mean():.2%}",
        }
    )

summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))

# %%
plt.plot(ttmet["time"])
plt.plot(settings["time"])
plt.axhline(target_times[target][0], color="r", linestyle="--", label="target start")
plt.axhline(target_times[target][1], color="g", linestyle="--", label="target end")
plt.legend()

# %%

settings_subset = settings[
    (settings["time"] >= target_times[target][0])
    & (settings["time"] <= target_times[target][1])
]

ttmet_subset = ttmet[
    (ttmet["time"] >= target_times[target][0])
    & (ttmet["time"] <= target_times[target][1])
]


servo_on_time_longest = find_longest_run(settings_subset["servo_mode"] == 1)
print(
    f"Longest servo_mode=1 run: length={servo_on_time_longest[0]}, start_idx={servo_on_time_longest[1]}, end_idx={servo_on_time_longest[2]}"
)

servo_times = [
    settings_subset["time"].iloc[servo_on_time_longest[1]],
    settings_subset["time"].iloc[servo_on_time_longest[2]],
]

plt.plot(settings_subset["time"], settings_subset["servo_mode"], label="servo_mode")

plt.axvline(servo_times[0], color="r", linestyle="--", label="servo on start")
plt.axvline(servo_times[1], color="g", linestyle="--", label="servo on end")
plt.legend()

# %%
ttmet_subset_servo_on = ttmet_subset[
    (ttmet_subset["time"] >= servo_times[0]) & (ttmet_subset["time"] <= servo_times[1])
]

# %%
plt.plot(ttmet_subset_servo_on["time"])

# %%
# plot_indicies = np.arange(int(ttmet_subset_servo_on.shape[0] * 0.5))

plt.plot(ttmet_subset_servo_on["time"], ttmet_subset_servo_on["tx"], label="mx")
plt.plot(ttmet_subset_servo_on["time"], ttmet_subset_servo_on["ty"], label="my")
plt.legend()
plt.xlabel("Time (UTC)")
plt.ylabel("Tip/Tilt (pixels)")
plt.ylim(-1.5, 4.5)

# %%
tx_demeaned = ttmet_subset_servo_on["tx"] - np.mean(ttmet_subset_servo_on["tx"])
ty_demeaned = ttmet_subset_servo_on["ty"] - np.mean(ttmet_subset_servo_on["ty"])

clip_vals = (-10, 10)
ty_demeaned = np.clip(ty_demeaned, *clip_vals)
tx_demeaned = np.clip(tx_demeaned, *clip_vals)

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

# %%
# rms = np.sqrt(np.mean(np.std(ttmet_subset_servo_on[["tx", "ty"]], axis=0) ** 2))
# instead of using std, use percentiles and convert to sigma
total_dev = np.sqrt(tx_demeaned**2 + ty_demeaned**2)
# use 90-10 percentiles and 2.563 conversion factor
p90 = np.percentile(total_dev, 90)
p10 = np.percentile(total_dev, 10)
rms = (p90 - p10) / 2.563


import astropy.units as u

# pixel_size = 24 * u.um / u.pix

# airy_ring_radius = 26.96 * u.um  # at J band
# # need to convert this to mas and get the plate scale in mas/pix to convert to pixels

# wavel = 1.25 * u.um  # J band
# D = 8.1 * u.m

# lam_over_D = (wavel / D).to(u.arcsec, equivalencies=u.dimensionless_angles())

# # one airy ring radius is equivalent to 1.22 * lam_over_D, so we can compute the plate scale as:
# plate_scale = 1.22 * lam_over_D.to(u.arcsec) / (airy_ring_radius / pixel_size)

# plate_scale.to(u.mas / u.pix)

plate_scale = 13 * u.mas / u.pix

rms_mas = (rms * u.pix * plate_scale).to(u.mas)

print(f"RMS in pixels: {rms:.3f} pix")
print(f"Plate scale: {plate_scale:.3f}")
print(f"RMS in mas: {rms_mas:.3f}")

# now need to find this in nm rms as tip/tilt modes
rms_rad = rms_mas / lam_over_D
rms_nm = (rms_rad * wavel / (2 * np.pi)).to(u.nm)

print(f"RMS in nm: {rms_nm:.3f}")

# %%
fps = 1 / np.median(
    np.diff(
        ttmet_subset_servo_on["time"].values.astype("datetime64[ns]").astype(float)
        / 1e9
    )
)
print(f"Estimated frame rate: {fps:.2f} Hz")

# PSD of each
plt.psd(mx_demeaned, Fs=fps, NFFT=256, label="mx")
plt.psd(my_demeaned, Fs=fps, NFFT=256, label="my")
plt.legend()

# %%
