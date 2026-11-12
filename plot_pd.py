"""Interactive gd_snr plot and PSD explorer for ft_performance logs.

Usage examples:
    python plot_pd.py /Users/mireland/data/mimir/20260914/ft_performance_20260914T221159.log \
        --start 2026-09-14T22:00:00 --end 2026-09-14T23:00:00 --length 4096
    python plot_pd.py /path/to/file.log --start-ut 1789436187 --end-ut 1789437300 --length 4096
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time


def parse_time(value):
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if text == "":
        return None
    if text.replace("-", "", 1).replace(".", "", 1).isdigit() or (
        text.startswith("-") and text[1:].replace(".", "", 1).isdigit()
    ):
        return float(text)

    for fmt in ("isot", "fits"):
        try:
            return Time(text, format=fmt, scale="utc").unix
        except Exception:
            pass

    try:
        return Time(datetime.fromisoformat(text), scale="utc").unix
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Could not parse time value: {value!r}") from exc


def read_ft_data(path: str | Path):
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    if df.shape[1] < 7:
        raise ValueError(f"Expected at least 7 columns in {path}, found {df.shape[1]}.")

    data = df.iloc[:, :].copy()
    data.columns = [f"col_{i}" for i in range(data.shape[1])]
    data["timestamp"] = pd.to_numeric(data["col_0"], errors="coerce")
    for i in range(1, 7):
        data[f"gd_snr_{i}"] = pd.to_numeric(data[f"col_{i}"], errors="coerce")
    return data.dropna(subset=["timestamp"]).reset_index(drop=True)


def select_window(data: pd.DataFrame, start_ut: float | None, end_ut: float | None):
    if start_ut is not None:
        data = data[data["timestamp"] >= start_ut]
    if end_ut is not None:
        data = data[data["timestamp"] <= end_ut]
    if data.empty:
        raise ValueError("No data rows fall within the requested time window.")
    return data


def _segment_for_click(data: pd.DataFrame, click_time: float, length: int):
    times = data["timestamp"].to_numpy(dtype=float)
    idx = int(np.argmin(np.abs(times - click_time)))
    start = min(idx, len(times) - 1)
    end = min(len(times), start + length)
    if end - start < 2:
        return None
    if start == len(times) - 1:
        start = max(0, len(times) - length)
        end = len(times)
    return data.iloc[start:end].copy(), start, end


def make_psd_window(segment: pd.DataFrame, click_time: float, length: int):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax2 = ax.twinx()

    start_dt = datetime.utcfromtimestamp(float(segment["col_0"].iloc[0])).strftime("%Y-%m-%dT%H:%M:%S")
    rcum_first_values = []

    for tel in [1, 2, 3, 4]:
        pd_tel = segment[f"col_{18 + tel}"].to_numpy(dtype=float) * 2.05
        dm_piston = segment[f"col_{26 + tel}"].to_numpy(dtype=float) * 7.0

        ps1 = np.abs(np.fft.rfft(pd_tel)) ** 2 / len(pd_tel) ** 2 * 2
        ps2 = np.abs(np.fft.rfft(dm_piston)) ** 2 / len(dm_piston) ** 2 * 2

        rcum_ps1 = np.cumsum(ps1[::-1])[::-1]
        rcum_ps2 = np.cumsum(ps2[::-1])[::-1]
        #This assumes 1 ms sample spacing, which is approximately correct. !!! Fix
        freqs = np.arange(len(ps1)) / len(pd_tel) * 1000.0

        rcum_first_values.append(float(rcum_ps1[0]))

        ax.semilogy(freqs, np.convolve(ps1 + ps2, np.ones(6) / 6, mode="same") * 3, color=f"C{tel}", linestyle=":")
        ax2.plot(freqs, rcum_ps1, color=f"C{tel}", label=f"AT{tel}")
        ax2.plot(freqs, rcum_ps2, color=f"C{tel}", linestyle="--")

    rms_value = 1000.0 * np.sqrt(np.mean(rcum_first_values))
    ax.text(0.7, 0.96, f"RMS: {rms_value:.1f} nm", transform=ax.transAxes, va="top", fontsize=12)
    ax.text(1.08, 0.5, r"Cumulative OPD ($\mu m^2$)", transform=ax.transAxes, rotation=90, va="center", ha="center")

    ax.axvline(click_time, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlim(0, 200)
    ax.set_ylim(1e-5, 1e-1)
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"Power spectral density ($\mu$m$^2$/Hz)")
    ax.set_title(f"Power Spectrum and Reverse Cumulative OPD at {start_dt}")
    ax.grid(True, alpha=0.3)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2)
    plt.show()


def plot_time_window(data: pd.DataFrame, start_ut: float | None, end_ut: float | None, length: int = 4096):
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

    x = data["timestamp"].to_numpy(dtype=float)
    for i in range(1, 7):
        ax.plot(x, data[f"gd_snr_{i}"].to_numpy(dtype=float), label=f"gd_snr_{i}", alpha=0.85)

    if start_ut is not None:
        ax.axvline(start_ut, color="k", linestyle="--", linewidth=1.0, alpha=0.7)
    if end_ut is not None:
        ax.axvline(end_ut, color="k", linestyle=":", linewidth=1.0, alpha=0.7)

    ax.set_title("gd_snr time series")
    ax.set_xlabel("Unix time (s)")
    ax.set_ylabel("gd_snr")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=3)

    selected_patch = None

    def on_click(event):
        nonlocal selected_patch
        if event.xdata is None:
            return
        segment = _segment_for_click(data, float(event.xdata), length)
        if segment is None:
            return
        seg, start_idx, end_idx = segment

        if selected_patch is not None:
            selected_patch.remove()
            selected_patch = None

        start_time = data.iloc[start_idx]["timestamp"]
        end_time = data.iloc[end_idx - 1]["timestamp"]
        selected_patch = ax.axvspan(start_time, end_time, color="gray", alpha=0.25)
        for a in fig.axes:
            a.axvline(start_time, color="tab:red", linestyle="--", linewidth=1.2)
        fig.canvas.draw_idle()
        make_psd_window(seg, start_time, length=length)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot gd_snr from ft_performance data and inspect PSD segments by clicking the time series.")
    parser.add_argument("infile", type=str, help="Path to the ft_performance log file")
    parser.add_argument("--length", type=int, default=4096, help="Length of the segment used for PSD analysis (default: 4096)")
    parser.add_argument("--start", type=str, default=None, help="Start date/time, e.g. 2026-09-14T22:00:00")
    parser.add_argument("--end", type=str, default=None, help="End date/time, e.g. 2026-09-14T23:00:00")
    parser.add_argument("--start-ut", type=float, default=None, help="Start unix timestamp")
    parser.add_argument("--end-ut", type=float, default=None, help="End unix timestamp")
    args = parser.parse_args()

    if (args.start is None and args.start_ut is None) or (args.end is None and args.end_ut is None):
        raise SystemExit("Provide both --start/--end or both --start-ut/--end-ut.")

    start_ut = parse_time(args.start_ut if args.start_ut is not None else args.start)
    end_ut = parse_time(args.end_ut if args.end_ut is not None else args.end)

    data = read_ft_data(args.infile)
    data = select_window(data, start_ut, end_ut)
    plot_time_window(data, start_ut, end_ut, length=args.length)


if __name__ == "__main__":
    main()
