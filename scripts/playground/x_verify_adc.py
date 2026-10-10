# %%
import datetime

import numpy as np
from heimdallrdr.DataBundler import DataBundler

import matplotlib.pyplot as plt
import pathlib

import astropy.units as u

# %%

base = pathlib.Path("/media/taras/T7/work/Asgard/Apr2026")


date_start = datetime.datetime(2026, 4, 2, tzinfo=datetime.timezone.utc)
time_start = "23:39:49"

date_end = datetime.datetime(2026, 4, 2, tzinfo=datetime.timezone.utc)
time_end = "23:41:54"

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
crop = "baldr1"

db = DataBundler(base)
times_used, frame_times, data = db.load_between(crop, start, end)

# %%
bad_pixel_mask = np.zeros(data.shape[1:], dtype=bool)
threshold = 10_000
# if a pixel exceeds the threshold in more than 10% of the frames, mark it as bad
bad_pixel_mask = np.sum(data > threshold, axis=0) > (0.1 * data.shape[0])

bad_pixel_mask[5, 11] = True

plt.imshow(bad_pixel_mask)

# set bad pixels to 1000.0
data[:, bad_pixel_mask] = 1000.0

# %%
# plt.imshow(data.mean(axis=0))
# plt.imshow(data[300_006], vmin=1000)

flux = data.sum((-1, -2))
maxes = data.max((-1, -2))

# %%
plt.plot(frame_times, flux)

time_subset = [
    datetime.datetime(2026, 4, 2, 23, 39, tzinfo=None),
    datetime.datetime(2026, 4, 2, 23, 41, 40, tzinfo=None),
]

plt.axvspan(time_subset[0], time_subset[1], color="red", alpha=0.3)

# %%
plt.plot(frame_times, maxes, alpha=0.5)

# moving average window of 100 frames
window_size = 201
moving_avg = np.convolve(maxes, np.ones(window_size) / window_size, mode="valid")
# times average should be shifted to match the moving average
times_avg = frame_times[window_size // 2 : -window_size // 2 + 1]
plt.plot(times_avg, moving_avg, color="red")

plt.axvspan(time_subset[0], time_subset[1], color="red", alpha=0.3)

# %%
data_subset = data[(frame_times > time_subset[0]) & (frame_times < time_subset[1])]
plt.imshow(data_subset.mean(axis=0), vmin=1000)
plt.colorbar()
# %%
pre_adc_mean = data_subset[0:1000].mean(0)
post_adc_mean = data_subset[-1000:-1].mean(0)

clims = (1000, max(pre_adc_mean.max(), post_adc_mean.max()))

plt.subplot(1, 2, 1)
plt.imshow(pre_adc_mean, vmin=1000, vmax=clims[1], cmap="inferno")
# text in top left corner of the image
plt.text(
    0.05,
    0.95,
    "Pre ADC",
    color="white",
    fontsize=12,
    transform=plt.gca().transAxes,
    verticalalignment="top",
)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(post_adc_mean, vmin=1000, vmax=clims[1], cmap="inferno")
plt.text(
    0.05,
    0.95,
    "Post ADC",
    color="white",
    fontsize=12,
    transform=plt.gca().transAxes,
    verticalalignment="top",
)
plt.axis("off")
# plt.savefig("adc_working.pdf", dpi=300, bbox_inches="tight")

# %%
# measure FWHM of the pre and post ADC images by fitting a 2D Gaussian to the images
from scipy.optimize import curve_fit

offset = 1000.0


def fit_2d_gaussian(data, x0, y0, amp, sig_x, sig_y, theta):
    """Fit a 2D Gaussian to the data."""
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    x, y = np.meshgrid(x, y)
    a = (np.cos(theta) ** 2) / (2 * sig_x**2) + (np.sin(theta) ** 2) / (2 * sig_y**2)
    b = -(np.sin(2 * theta)) / (4 * sig_x**2) + (np.sin(2 * theta)) / (4 * sig_y**2)
    c = (np.sin(theta) ** 2) / (2 * sig_x**2) + (np.cos(theta) ** 2) / (2 * sig_y**2)
    g = offset + amp * np.exp(
        -(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2))
    )
    return g.ravel()


def measure_fwhm(data):
    """Measure the FWHM of the data by fitting a 2D Gaussian."""
    # initial guess for the parameters
    x0 = data.shape[1] / 2
    y0 = data.shape[0] / 2
    amp = data.max() - data.min()
    sig_x = 5
    sig_y = 5
    theta = 0

    p0 = [x0, y0, amp, sig_x, sig_y, theta]

    # fit the data
    popt, pcov = curve_fit(fit_2d_gaussian, data, data.ravel(), p0=p0, maxfev=10000)

    fwhm_x = 2 * np.sqrt(2 * np.log(2)) * popt[3]
    fwhm_y = 2 * np.sqrt(2 * np.log(2)) * popt[4]
    return fwhm_x, fwhm_y, popt


plt.figure()
plt.subplot(1, 2, 1)
fwhm_pre_x, fwhm_pre_y, popt_pre = measure_fwhm(pre_adc_mean)
plt.imshow(pre_adc_mean, vmin=offset, vmax=clims[1], cmap="inferno")

x, y = np.meshgrid(np.arange(pre_adc_mean.shape[1]), np.arange(pre_adc_mean.shape[0]))
plt.contour(
    x,
    y,
    fit_2d_gaussian(x, *popt_pre).reshape(pre_adc_mean.shape),
    colors="white",
    levels=[offset + 0.5 * popt_pre[2]],  # contour at half max
    alpha=0.5,
)
# plt.title(f"Pre ADC FWHM: {fwhm_pre_x:.2f} x {fwhm_pre_y:.2f} pixels")
plt.axis("off")
plt.text(
    0.05,
    0.95,
    f"Pre ADC \nFWHM: {fwhm_pre_x:.2f} x {fwhm_pre_y:.2f} px",
    color="white",
    fontsize=12,
    transform=plt.gca().transAxes,
    verticalalignment="top",
)

plt.subplot(1, 2, 2)
fwhm_post_x, fwhm_post_y, popt_post = measure_fwhm(post_adc_mean)
plt.imshow(post_adc_mean, vmin=offset, vmax=clims[1], cmap="inferno")
x, y = np.meshgrid(np.arange(post_adc_mean.shape[1]), np.arange(post_adc_mean.shape[0]))
plt.contour(
    x,
    y,
    fit_2d_gaussian(x, *popt_post).reshape(post_adc_mean.shape),
    colors="white",
    levels=[offset + 0.5 * popt_post[2]],  # contour at half max
    alpha=0.5,
)
# plt.title(f"Post ADC FWHM: {fwhm_post_x:.2f} x {fwhm_post_y:.2f} pixels")
plt.axis("off")
plt.text(
    0.05,
    0.95,
    f"Post ADC \nFWHM: {fwhm_post_x:.2f} x {fwhm_post_y:.2f} px",
    color="white",
    fontsize=12,
    transform=plt.gca().transAxes,
    verticalalignment="top",
)


# %%
