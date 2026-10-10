# %%
import datetime

# from heimdallrdr.DataBundler import DataBundler

import matplotlib.pyplot as plt
import pathlib

# from heimdallrdr.FtSettingsBundler import FTSettingsBundler
# from heimdallrdr.FTPerformanceBundler import FTPerformanceBundler

import FTPerformanceBundler

# %%

base = pathlib.Path("/media/taras/T7/work/Asgard/Apr2026")
# base = pathlib.Path("~/Desktop").expanduser()
# base = pathlib.Path(__file__).parent.parent.parent / "example_files"


date_start = datetime.datetime(2026, 4, 11, tzinfo=datetime.timezone.utc)
time_start = "20:40:14"

date_end = datetime.datetime(2026, 4, 12, tzinfo=datetime.timezone.utc)
time_end = "09:30:14"

start = date_start + datetime.timedelta(
    hours=int(time_start.split(":")[0]),
    minutes=int(time_start.split(":")[1]),
    seconds=int(time_start.split(":")[2]),
)
end = date_end + datetime.timedelta(
    hours=int(time_end.split(":")[0]),
    minutes=int(time_end.split(":")[1]),
    seconds=int(time_end.split(":")[2]),
)


# %%
ple = FTPerformanceBundler.PerformanceLogExtractor(base)

rows = ple.extract_rows(start, end)

# %%
cols = [f"gd_snr_{i}" for i in range(1, 7)]
# cols = [f"gd_bl_{i}" for i in range(1,5)]

for col in cols:
    plt.plot(rows["timestamp"], rows[col], label=col, alpha=0.5)

plt.legend()
plt.ylim(0, 100)


# %%
# moving average
window_size = 1001

plt.figure(figsize=(12, 6))
for col in cols:
    plt.plot(
        rows["timestamp"],
        rows[col].rolling(window=window_size, center=True).mean(),
        label=col,
    )
plt.legend()
plt.ylim(0, 100)

# %%
start = [
    datetime.datetime(2026, 4, 12, tzinfo=datetime.timezone.utc),
    "02:36:25",
]

end = [
    datetime.datetime(2026, 4, 12, tzinfo=datetime.timezone.utc),
    "02:55:14",
]

rows = ple.extract_rows(
    start[0] + datetime.timedelta(
        hours=int(start[1].split(":")[0]),
        minutes=int(start[1].split(":")[1]),
        seconds=int(start[1].split(":")[2]),
    ),
    end[0] + datetime.timedelta(
        hours=int(end[1].split(":")[0]),
        minutes=int(end[1].split(":")[1]),
        seconds=int(end[1].split(":")[2]),
    ),
)  


# %%
plt.figure(figsize=(12, 6))
for col in cols:
    plt.plot(
        rows["timestamp"],
        rows[col],
        label=col,
    )
plt.legend()

# %%

# ft_settings_base = base
# ft_settings_bundler = FTSettingsBundler(ft_settings_base)

# ft_settings_start = start
# ft_settings_end = end

# ft_settings_df, ft_settings_frac = ft_settings_bundler.get_settings_between(
#     ft_settings_start,
#     ft_settings_end,
# )

# print(
#     f"Loaded {ft_settings_df.shape[0]} ft settings rows from "
#     f"{ft_settings_start} to {ft_settings_end} "
#     f"(coverage {ft_settings_frac * 100:.1f}%)."
# )

# ft_settings_df.head()

# %%
# plt.plot(ft_settings_df["time"], ft_settings_df["gd_threshold"])

# %%

ft_perf_bundler = FTPerformanceBundler.FTPerformanceBundler(base)

ft_perf_df, ft_perf_frac = ft_perf_bundler.get_performance_between(
    start,
    end,
)

print(
    f"Loaded {ft_perf_df.shape[0]} ft performance rows from "
    f"{start} to {end} "
    f"(coverage {ft_perf_frac * 100:.1f}%)."
)

ft_perf_df.head()

# %%
plt.plot(ft_perf_df["time"], ft_perf_df["gd_snr"], label="gd_snr")
# plt.plot(ft_perf_df["time"], ft_perf_df["pd_snr"], label="pd_snr")
plt.legend()
plt.figure()
# plt.plot(ft_perf_df["time"], ft_perf_df["gd_threshold"], label="gd_threshold")
# %%
crop = "hei_k2"

db = DataBundler(base)
times_used, frame_times, data = db.load_between(crop, start, end)

# %%
plt.imshow(data[10_000])
