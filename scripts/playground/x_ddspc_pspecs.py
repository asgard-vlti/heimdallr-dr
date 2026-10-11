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


def detect_pd_tel_3_bump(rows, values, discard_samples):
    sampling_hz = 1.0 / rows["time"].diff().dt.total_seconds().iloc[1:].median()
    smooth_samples = max(3, round(0.5 * sampling_hz))
    sustained_samples = max(1, round(0.25 * sampling_hz))
    smoothed = (
        pd.Series(values[:, 2])
        .rolling(smooth_samples, center=True, min_periods=smooth_samples // 2)
        .mean()
        .to_numpy()
    )
    baseline_samples = smoothed[discard_samples : len(smoothed) // 2]
    baseline = onp.nanmedian(baseline_samples)
    baseline_mad = onp.nanmedian(onp.abs(baseline_samples - baseline))
    threshold = max(0.01, 6 * 1.4826 * baseline_mad)
    away_from_baseline = onp.abs(smoothed - baseline) > threshold
    sustained = (
        pd.Series(away_from_baseline)
        .rolling(sustained_samples, min_periods=sustained_samples)
        .sum()
        .to_numpy()
    )
    search_start = max(discard_samples, len(smoothed) // 2)
    detected = onp.flatnonzero(
        (onp.arange(len(smoothed)) >= search_start + sustained_samples - 1)
        & (sustained >= sustained_samples)
    )
    onset = int(detected[0] - sustained_samples + 1) if len(detected) else None
    pre_bump_rms_nm = (
        mean_pd_tel_rms_nm(values[:onset], discard_samples)
        if onset is not None
        else None
    )
    return {
        "onset": onset,
        "smoothed": smoothed,
        "pre_bump_rms_nm": pre_bump_rms_nm,
    }


def plot_pd_tel_3_bump(ax, rows, bump):
    ax.plot(
        rows["time"],
        bump["smoothed"],
        color="black",
        linewidth=1.2,
        label="pd_tel_3 (0.5 s mean)",
    )
    if bump["onset"] is not None:
        ax.axvline(
            rows["time"].iloc[bump["onset"]],
            color="magenta",
            linestyle="--",
            linewidth=1.4,
            label="Detected bump onset",
        )


def mode_1_rms_label(full_rms_nm, bump):
    if bump["onset"] is None:
        return f"Full mean RMS {full_rms_nm:.0f} nm; no sustained pd_tel_3 bump detected"
    return (
        f"Full mean RMS {full_rms_nm:.0f} nm; "
        f"before bump {bump['pre_bump_rms_nm']:.0f} nm"
    )


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
    performance,
    servo_mode,
    run_bounds,
    min_samples,
    discard_samples,
    discard_reason,
    bump_results=None,
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
        bump = bump_results.get(number) if bump_results is not None else None
        if bump is not None:
            plot_pd_tel_3_bump(ax, rows, bump)
        if onp.isfinite(mean_rms_nm):
            run_metrics.append((mean_rms_nm, number, start, end))
        start_local = rows["time"].iloc[0].tz_convert(local_tz)
        end_local = rows["time"].iloc[-1].tz_convert(local_tz)
        rms_label = (
            mode_1_rms_label(mean_rms_nm, bump)
            if bump is not None
            else f"Mean RMS: {mean_rms_nm:.0f} nm"
        )
        ax.set_title(
            f"{number}. {start_local:%H:%M:%S}–{end_local:%H:%M:%S} "
            f"({len(rows):,} samples)\n{rms_label} "
            f"(after {discard_samples:,} {discard_reason} samples)",
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

def plot_pd_tel_psd(ax, rows, discard_samples, title):
    psd_rows = rows.iloc[discard_samples:]
    pd_tel_psd = ft_perf.measurement_array(psd_rows, "pd_tel")
    sample_intervals = psd_rows["time"].diff().dt.total_seconds().iloc[1:]
    sampling_hz = 1.0 / sample_intervals.median()
    nfft = min(4096, 2 ** int(onp.log2(len(psd_rows))))

    plt.sca(ax)
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
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB K1 waves²/Hz)")
    ax.legend(ncol=4)
    ax.grid(alpha=0.25)
    return sampling_hz


# %%
def plot_best_run_psds(
    performance,
    run_metrics,
    servo_mode,
    discard_samples,
    discard_reason,
    bump_results=None,
):
    best_five = sorted(run_metrics)[:5]
    if not best_five:
        raise ValueError(f"No servo_mode={servo_mode} runs with finite pd_tel RMS")

    for rank, (mean_rms_nm, run_number, start, end) in enumerate(best_five, start=1):
        rows = performance.iloc[start : end + 1]
        fig, (ax_time, ax_psd) = plt.subplots(
            2, 1, figsize=(14, 8), constrained_layout=True
        )
        plot_pd_tel(ax_time, rows)
        bump = bump_results.get(run_number) if bump_results is not None else None
        if bump is not None:
            plot_pd_tel_3_bump(ax_time, rows, bump)
        start_local = rows["time"].iloc[0].tz_convert(local_tz)
        end_local = rows["time"].iloc[-1].tz_convert(local_tz)
        ax_time.set_title(
            f"Run {run_number}: {start_local:%H:%M:%S}–{end_local:%H:%M:%S} "
            f"({len(rows):,} samples)"
        )
        ax_time.legend(ncol=4)

        plot_pd_tel_psd(
            ax_psd,
            rows,
            discard_samples,
            f"PSD after discarding {discard_samples:,} {discard_reason} samples",
        )
        rms_label = (
            mode_1_rms_label(mean_rms_nm, bump)
            if bump is not None
            else f"Mean pd_tel RMS {mean_rms_nm:.0f} nm"
        )
        fig.suptitle(
            f"Servo mode {servo_mode} — best #{rank}: {rms_label} "
            f"(after {discard_samples:,} {discard_reason} samples)"
        )
        print(
            f"Servo mode {servo_mode} best #{rank}: run {run_number}, "
            f"{rms_label} after {discard_samples:,} {discard_reason} samples"
        )
        if bump is not None and bump["onset"] is not None:
            print(
                f"  Bump onset: "
                f"{rows['time'].iloc[bump['onset']].tz_convert(local_tz)}"
            )

    plt.show()
    return best_five


best_five = plot_best_run_psds(
    performance_zoom, run_metrics, 5, ddspc_discard_samples, "exploration"
)

# %%
mode_1_bounds = find_mode_runs(performance_zoom, 1, min_samples)
mode_1_bumps = {}
for number, (start, end) in enumerate(mode_1_bounds, start=1):
    rows = performance_zoom.iloc[start : end + 1]
    values = ft_perf.measurement_array(rows, "pd_tel")
    mode_1_bumps[number] = detect_pd_tel_3_bump(
        rows, values, mode_1_discard_samples
    )

mode_1_metrics = plot_mode_mega(
    performance_zoom,
    1,
    mode_1_bounds,
    min_samples,
    mode_1_discard_samples,
    "loop-closing",
    mode_1_bumps,
)

# %%
best_mode_1 = plot_best_run_psds(
    performance_zoom,
    mode_1_metrics,
    1,
    mode_1_discard_samples,
    "loop-closing",
    mode_1_bumps,
)

# %%
mode_4_bounds = [
    (start, end)
    for start, end in find_mode_runs(performance_zoom, 4, 1)
    if start > 0 and end < len(performance_zoom) - 1
]
if not mode_4_bounds:
    raise ValueError("No interior servo_mode=4 run in the selected interval")

mode_4_start, mode_4_end = max(
    mode_4_bounds,
    key=lambda bounds: (
        performance_zoom["time"].iloc[bounds[1]]
        - performance_zoom["time"].iloc[bounds[0]]
    ),
)
mode_4_run = performance_zoom.iloc[mode_4_start : mode_4_end + 1]
mode_4_values = ft_perf.measurement_array(mode_4_run, "pd_tel")
mode_4_rms_nm = mean_pd_tel_rms_nm(mode_4_values, 0)
mode_4_local_start = mode_4_run["time"].iloc[0].tz_convert(local_tz)
mode_4_local_end = mode_4_run["time"].iloc[-1].tz_convert(local_tz)

fig, (ax_time, ax_psd) = plt.subplots(
    2, 1, figsize=(14, 8), constrained_layout=True
)
plot_pd_tel(ax_time, mode_4_run)
ax_time.set_title(
    f"Longest interior servo_mode=4 run: "
    f"{mode_4_local_start:%H:%M:%S}–{mode_4_local_end:%H:%M:%S} "
    f"({len(mode_4_run):,} samples)"
)
ax_time.legend(ncol=4)
mode_4_sampling_hz = plot_pd_tel_psd(
    ax_psd, mode_4_run, 0, "PSD of the full servo-off run"
)
fig.suptitle(
    f"Servo mode 4 — mean pd_tel RMS over all four telescopes: "
    f"{mode_4_rms_nm:.0f} nm (full run, no samples discarded)"
)
print(
    f"Longest interior servo_mode=4 run: {len(mode_4_run):,} samples, "
    f"{mode_4_local_start} to {mode_4_local_end}; "
    f"mean pd_tel RMS {mode_4_rms_nm:.0f} nm, "
    f"PSD sampling rate {mode_4_sampling_hz:.1f} Hz"
)
plt.show()

# %%
