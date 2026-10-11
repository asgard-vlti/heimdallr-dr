# %%
import datetime
from pathlib import Path

import heimdallrdr as hdr
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd
from heimdallrdr.utils import find_longest_run

day_start = datetime.datetime(2026, 10, 10, tzinfo=datetime.timezone.utc)
day_end = day_start + datetime.timedelta(days=1)
data_dir = Path(__file__).resolve().parents[3] / "data" / "20261010_comp"
# K1 wavelength in dcs/heimdallr/def.toml; adjust if the observing config differed.
k1_wavelength_um = 2.1

ft = hdr.FTSettingsBundler(data_dir)


# %%
settings, _ = ft.get_settings_between(day_start, day_end)
settings = settings.loc[settings["time"] < day_end].copy()
if settings.empty:
    raise FileNotFoundError(f"No FT settings found for 2026-10-10 in {data_dir}")

files = [
    sf
    for sf in ft.files
    if sf.start_time is not None
    and sf.end_time is not None
    and sf.end_time >= day_start
    and sf.start_time < day_end
]
gap_rows = []
for previous, following in zip(files, files[1:]):
    gap_start = max(previous.end_time, day_start)
    gap_end = min(following.start_time, day_end)
    if gap_start < gap_end:
        gap_rows.append({"time": gap_start + (gap_end - gap_start) / 2})

settings_plot = (
    pd.concat([settings, pd.DataFrame(gap_rows, columns=["time"])], ignore_index=True)
    .sort_values("time")
    .reset_index(drop=True)
)
print(f"Loaded {len(settings)} FT settings rows with {len(gap_rows)} file gaps.")


# %%
setting_columns = [
    column for column in settings.columns if column not in {"time", "servo_mode"}
]
fig, axes = plt.subplots(
    len(setting_columns),
    1,
    figsize=(14, 1.8 * len(setting_columns)),
    sharex=True,
    constrained_layout=True,
)
for ax, column in zip(axes, setting_columns):
    ax.plot(settings_plot["time"], settings_plot[column], linewidth=0.8)
    ax.set_ylabel(column)
    ax.grid(alpha=0.25)

axes[-1].set_xlim(day_start, day_end)
axes[-1].xaxis.set_major_locator(
    mdates.HourLocator(interval=2, tz=datetime.timezone.utc)
)
axes[-1].xaxis.set_major_formatter(
    mdates.DateFormatter("%H:%M", tz=datetime.timezone.utc)
)
axes[-1].set_xlabel("Time (UTC)")
fig.suptitle("FT settings — 2026-10-10")
plt.show()


# %%
local_tz = datetime.timezone(datetime.timedelta(hours=-3))
zoom_start = datetime.datetime(2026, 10, 10, 14, 19, 29, tzinfo=local_tz)
zoom_end = datetime.datetime(2026, 10, 10, 14, 46, 43, tzinfo=local_tz)

fig, ax = plt.subplots(figsize=(14, 3), constrained_layout=True)
ax.plot(settings_plot["time"], settings_plot["servo_mode"], linewidth=0.8)
ax.set_xlim(zoom_start, zoom_end)
ax.set_yticks(sorted(settings["servo_mode"].dropna().unique()))
ax.set_ylabel("Servo mode")
ax.set_xlabel("Time (UTC−3)")
ax.set_title("FT servo mode — 2026-10-10")
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5, tz=local_tz))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=local_tz))
ax.grid(alpha=0.25)
plt.show()

# %%
settings_zoom = settings_plot.loc[
    settings_plot["time"].between(zoom_start, zoom_end)
].reset_index(drop=True)
run_length, first, last = find_longest_run(settings_zoom["servo_mode"].eq(5))
if run_length == 0:
    raise ValueError("No servo_mode=5 run in the selected time window")

coarse_start = settings_zoom["time"].iloc[first]
coarse_end = settings_zoom["time"].iloc[last]
print(
    f"Longest servo_mode=5 run: {run_length} settings rows, "
    f"{coarse_start.tz_convert(local_tz)} to {coarse_end.tz_convert(local_tz)}"
)


# %%
ft_perf = hdr.FTPerformanceBundler(data_dir)
padding = datetime.timedelta(seconds=2)
performance, _ = ft_perf.get_performance_between(
    coarse_start - padding, coarse_end + padding
)
if performance.empty:
    raise ValueError(f"No FT performance rows near {coarse_start} to {coarse_end}")
if "servo_mode" not in performance.columns:
    raise KeyError("The relevant FT performance log has no servo_mode column")

mode_5 = pd.to_numeric(performance["servo_mode"], errors="coerce").eq(5)
perf_length, perf_first, perf_last = find_longest_run(mode_5)
if perf_length == 0:
    raise ValueError("No servo_mode=5 samples in the relevant FT performance log")
if perf_first == 0 or perf_last == len(performance) - 1:
    raise ValueError("Performance run reaches the loaded interval edge; increase padding")

performance_run = performance.iloc[perf_first : perf_last + 1].copy()
run_start = performance_run["time"].iloc[0]
run_end = performance_run["time"].iloc[-1]

relevant_logs = [
    sf.path.name
    for sf in ft_perf.files
    if sf.start_time is not None
    and sf.end_time is not None
    and sf.end_time >= coarse_start - padding
    and sf.start_time <= coarse_end + padding
]
print(
    f"Refined servo_mode=5 run: {perf_length} performance rows, "
    f"{run_start.tz_convert(local_tz)} to {run_end.tz_convert(local_tz)} "
    f"from {', '.join(relevant_logs)}"
)

def plot_pd_tel(ax, rows):
    values = ft_perf.measurement_array(rows, "pd_tel")
    for telescope in range(values.shape[1]):
        ax.plot(
            rows["time"],
            values[:, telescope],
            label=f"pd_tel_{telescope + 1}",
            linewidth=0.8,
        )

    ax.set_xlim(rows["time"].iloc[0], rows["time"].iloc[-1])
    ax.set_xlabel("Time (UTC−3)")
    ax.set_ylabel("pd_tel (K1 waves)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz=local_tz))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=local_tz))
    ax.grid(alpha=0.25)
    return values


def mean_pd_tel_rms_nm(values, discard_samples):
    kept_values = values[discard_samples:]
    if len(kept_values) == 0:
        raise ValueError("No pd_tel samples remain after discarding the initial samples")
    telescope_rms_waves = onp.sqrt(onp.nanmean(onp.square(kept_values), axis=0))
    return onp.nanmean(telescope_rms_waves) * k1_wavelength_um * 1_000


fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
plot_pd_tel(ax, performance_run)
ax.set_title("PD telescope values during the longest servo_mode=5 run")
ax.legend()
plt.show()

# %%
performance_zoom, _ = ft_perf.get_performance_between(zoom_start, zoom_end)
if performance_zoom.empty or "servo_mode" not in performance_zoom.columns:
    raise ValueError("No servo mode data in the selected performance interval")

min_samples = 6000
ddspc_discard_samples = 5_000
mode_1_discard_samples = 100


def find_mode_runs(performance, servo_mode, min_samples):
    in_mode = pd.to_numeric(performance["servo_mode"], errors="coerce").eq(
        servo_mode
    )
    time_gap = performance["time"].diff().gt(pd.Timedelta(seconds=1))
    run_starts = performance.index[
        in_mode & (~in_mode.shift(fill_value=False) | time_gap)
    ]
    run_ends = performance.index[
        in_mode
        & (~in_mode.shift(-1, fill_value=False) | time_gap.shift(-1, fill_value=False))
    ]
    return [
        (start, end)
        for start, end in zip(run_starts, run_ends)
        if end - start + 1 >= min_samples
    ]


def plot_mode_mega(
    performance, servo_mode, run_bounds, min_samples, discard_samples, discard_reason
):
    if not run_bounds:
        raise ValueError(
            f"No servo_mode={servo_mode} runs with at least {min_samples} samples"
        )

    n_columns = 2
    n_rows = (len(run_bounds) + n_columns - 1) // n_columns
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(16, 2.8 * n_rows),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    run_metrics = []
    for number, ((start, end), ax) in enumerate(zip(run_bounds, axes.flat), start=1):
        rows = performance.iloc[start : end + 1]
        values = plot_pd_tel(ax, rows)
        mean_rms_nm = mean_pd_tel_rms_nm(values, discard_samples)
        if onp.isfinite(mean_rms_nm):
            run_metrics.append((mean_rms_nm, number, start, end))
        start_local = rows["time"].iloc[0].tz_convert(local_tz)
        end_local = rows["time"].iloc[-1].tz_convert(local_tz)
        ax.set_title(
            f"{number}. {start_local:%H:%M:%S}–{end_local:%H:%M:%S} "
            f"({len(rows):,} samples)\nMean RMS after {discard_samples:,} "
            f"{discard_reason} samples: {mean_rms_nm:.0f} nm",
            fontsize=10,
        )
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)

    for ax in list(axes.flat)[len(run_bounds) :]:
        ax.set_visible(False)

    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle(
        f"pd_tel during servo_mode={servo_mode} runs with at least "
        f"{min_samples:,} samples — 2026-10-10 (UTC−3)"
    )
    print(
        f"Plotted {len(run_bounds)} servo_mode={servo_mode} runs "
        f"with at least {min_samples} samples; RMS excludes the first "
        f"{discard_samples:,} {discard_reason} samples"
    )
    plt.show()
    return run_metrics


run_bounds = find_mode_runs(performance_zoom, 5, min_samples)
run_metrics = plot_mode_mega(
    performance_zoom,
    5,
    run_bounds,
    min_samples,
    ddspc_discard_samples,
    "exploration",
)

# %%
def plot_best_run_psds(
    performance, run_metrics, servo_mode, discard_samples, discard_reason
):
    best_five = sorted(run_metrics)[:5]
    if not best_five:
        raise ValueError(f"No servo_mode={servo_mode} runs with finite pd_tel RMS")

    for rank, (mean_rms_nm, run_number, start, end) in enumerate(best_five, start=1):
        rows = performance.iloc[start : end + 1]
        psd_rows = rows.iloc[discard_samples:]
        pd_tel_psd = ft_perf.measurement_array(psd_rows, "pd_tel")
        sample_intervals = psd_rows["time"].diff().dt.total_seconds().iloc[1:]
        sampling_hz = 1.0 / sample_intervals.median()
        nfft = min(4096, 2 ** int(onp.log2(len(psd_rows))))

        fig, (ax_time, ax_psd) = plt.subplots(
            2, 1, figsize=(14, 8), constrained_layout=True
        )
        plot_pd_tel(ax_time, rows)
        start_local = rows["time"].iloc[0].tz_convert(local_tz)
        end_local = rows["time"].iloc[-1].tz_convert(local_tz)
        ax_time.set_title(
            f"Run {run_number}: {start_local:%H:%M:%S}–{end_local:%H:%M:%S} "
            f"({len(rows):,} samples)"
        )
        ax_time.legend(ncol=4)

        plt.sca(ax_psd)
        for telescope in range(pd_tel_psd.shape[1]):
            plt.psd(
                pd_tel_psd[:, telescope],
                NFFT=nfft,
                Fs=sampling_hz,
                noverlap=nfft // 2,
                detrend="mean",
                scale_by_freq=True,
                label=f"pd_tel_{telescope + 1}",
            )
        ax_psd.set_title(
            f"PSD after discarding {discard_samples:,} {discard_reason} samples"
        )
        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_ylabel("PSD (dB K1 waves²/Hz)")
        ax_psd.legend(ncol=4)
        ax_psd.grid(alpha=0.25)
        fig.suptitle(
            f"Servo mode {servo_mode} — best #{rank}: "
            f"mean pd_tel RMS after {discard_samples:,} "
            f"{discard_reason} samples {mean_rms_nm:.0f} nm"
        )
        print(
            f"Servo mode {servo_mode} best #{rank}: run {run_number}, "
            f"mean RMS after {discard_samples:,} {discard_reason} "
            f"samples {mean_rms_nm:.0f} nm"
        )

    plt.show()
    return best_five


best_five = plot_best_run_psds(
    performance_zoom, run_metrics, 5, ddspc_discard_samples, "exploration"
)

# %%
mode_1_bounds = find_mode_runs(performance_zoom, 1, min_samples)
mode_1_metrics = plot_mode_mega(
    performance_zoom,
    1,
    mode_1_bounds,
    min_samples,
    mode_1_discard_samples,
    "loop-closing",
)

# %%
best_mode_1 = plot_best_run_psds(
    performance_zoom, mode_1_metrics, 1, mode_1_discard_samples, "loop-closing"
)

# %%
