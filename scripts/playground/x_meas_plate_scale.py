# %%
import datetime

import numpy as np
from heimdallrdr.DataBundler import DataBundler

import matplotlib.pyplot as plt
import pathlib

import astropy.units as u

# %%

base = pathlib.Path("/media/taras/T7/work/Asgard/Apr2026")


date_start = datetime.datetime(2026, 4, 2)
time_start = "01:04:48"

date_end = datetime.datetime(2026, 4, 2)
time_end = "01:19:06"

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


# crop = "hei_k1"
crop = "baldr2"

db = DataBundler(base)
times_used, frame_times, data = db.load_between(crop, start, end)

# %%
bad_pixel_mask = np.zeros(data.shape[1:], dtype=bool)
threshold = 20_000
# if a pixel exceeds the threshold in more than 10% of the frames, mark it as bad
bad_pixel_mask = np.sum(data > threshold, axis=0) > (0.1 * data.shape[0])
plt.imshow(bad_pixel_mask)

# %%
plt.imshow(data.mean(axis=0)*(~bad_pixel_mask), vmin=1000)
# plt.imshow(data[300_000]*(~bad_pixel_mask), vmin=1000)

# beam1 by eye
primary = np.array([14.0,18.5])
secondary = np.array([23.0,14.8])

# beam2 by eye
primary = np.array([14.0,18.5])
secondary = np.array([23.0,14.8])

# plt.scatter(primary[0], primary[1], color="red", marker="x")
# plt.scatter(secondary[0], secondary[1], color="red", marker="x")
# plt.imshow(data[100])

pix_distance = np.linalg.norm(primary - secondary)
print(f"Pixel distance: {pix_distance:.2f} pixels")

# %%
expected_sep = 118 * u.mas

plate_scale = expected_sep / (pix_distance * u.pixel)
print(f"Plate scale: {plate_scale:.2f}")

# %%
flux = ((data-1000.0)*(~bad_pixel_mask)).sum((-1, -2))
plt.plot(frame_times, flux)

# %%
# centered moving average
window_size = 51
moving_avg = np.convolve(flux, np.ones(window_size)/window_size, mode='same')
plt.plot(frame_times, moving_avg)

# %%
plt.plot(np.diff(moving_avg)[100:-100])