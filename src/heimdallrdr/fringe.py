"""Offline K1 phase delay in baseline order 12, 13, 14, 23, 24, 34.

Coordinates are signed (u, v) Fourier pixels; image arrays are (y, x).
All phases are in K1 wavelengths. No temporal phase unwrapping is applied.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as onp
from scipy.optimize import least_squares

from ._fits import index_fits, read_cube_batches

BASELINES = ("12", "13", "14", "23", "24", "34")


@dataclass
class SplodgeFit:
    mean_power: onp.ndarray
    initial_centres: onp.ndarray
    centres: onp.ndarray
    patch_coordinates: onp.ndarray
    patches: onp.ndarray
    models: onp.ndarray
    masks: onp.ndarray
    residuals: onp.ndarray
    parameters: onp.ndarray
    success: onp.ndarray
    at_bound: onp.ndarray
    messages: tuple[str, ...]
    frame_count: int = 0
    central_radius: float = 4.0


@dataclass
class PhasorResult:
    phasors: onp.ndarray
    noise_power: onp.ndarray
    valid_frames: onp.ndarray


@dataclass
class PhaseDelayResult:
    pd_snr: onp.ndarray
    pd_baseline: onp.ndarray
    pd_tel: onp.ndarray
    valid_baselines: onp.ndarray
    valid_noise: onp.ndarray
    connected: onp.ndarray
    valid: onp.ndarray


@dataclass
class ExtractionResult(PhaseDelayResult):
    times: onp.ndarray
    source_files: tuple[Path, ...]
    file_indices: onp.ndarray
    frame_indices: onp.ndarray
    fps: onp.ndarray
    phasors: onp.ndarray
    noise_power: onp.ndarray
    valid_frames: onp.ndarray
    fit: SplodgeFit


class SplodgeFitError(ValueError):
    """A fit needs corrected seeds; diagnostics remain available as ``fit``."""

    def __init__(self, fit):
        self.fit = fit
        failed = ", ".join(name for name, ok, bound in
                           zip(BASELINES, fit.success, fit.at_bound) if not ok or bound)
        super().__init__(f"Splodge fits failed or reached bounds for {failed}; "
                         "correct initial_centres. Diagnostics are in error.fit.")


def fit_splodge_centres(
    mean_power, initial_centres, *, patch_size=7, central_radius=4.0,
    max_displacement=2.0, width_bounds=(0.3, 2.5),
) -> SplodgeFit:
    """Fit six elliptical Gaussians plus planes to an unshifted mean power FFT.

    Returns diagnostics even for failures; extraction rejects failed/bounded fits.
    Patch coordinates and fitted centres are continuous signed (u, v) pixels,
    locally unrolled across Fourier boundaries. Widths are Gaussian sigmas.
    ``parameters`` columns are amplitude, du, dv, sigma1, sigma2, angle (radians),
    background, slope_u, slope_v. du/dv are relative to initial_centres.
    Displacements reaching max_displacement in Euclidean distance need new seeds.
    The robust soft-L1 residual scale is 5% of the unmasked patch maximum.
    """
    power = onp.asarray(mean_power, dtype=float)
    seeds = _centres(initial_centres)
    if power.ndim != 2 or not onp.all(onp.isfinite(power)) or onp.any(power < 0):
        raise ValueError("mean_power must be a finite, nonnegative 2D unshifted spectrum")
    if not isinstance(patch_size, (int, onp.integer)) or patch_size < 5 or patch_size % 2 != 1 or patch_size > min(power.shape):
        raise ValueError("patch_size must be odd, >= 5, and no larger than the image")
    if not onp.isfinite([central_radius, max_displacement, *width_bounds]).all() or central_radius < 0 or max_displacement <= 0 or not 0 < width_bounds[0] < width_bounds[1]:
        raise ValueError("Invalid mask radius, displacement or width bounds")
    ny, nx = power.shape
    dv, du = onp.mgrid[-patch_size//2+1:patch_size//2+1,
                       -patch_size//2+1:patch_size//2+1]
    coordinates, patches, models, masks, params, centres = [], [], [], [], [], []
    successes, bounds, messages = [], [], []
    for seed in seeds:
        u = du + onp.rint(seed[0])
        v = dv + onp.rint(seed[1])
        x, y = u - seed[0], v - seed[1]
        data = power[v.astype(int) % ny, u.astype(int) % nx]
        radial_u = (u + nx/2) % nx - nx/2
        radial_v = (v + ny/2) % ny - ny/2
        mask = onp.hypot(radial_u, radial_v) > central_radius
        model = onp.full_like(data, onp.nan)
        par = onp.full(9, onp.nan)
        centre = onp.full(2, onp.nan)
        success, at_bound, message = False, False, "Too few unmasked patch pixels"
        if mask.sum() >= 12 and onp.ptp(data[mask]) > 0:
            scale = max(float(data[mask].max()), onp.finfo(float).tiny)
            normalized = data / scale
            background = float(onp.median(normalized[mask]))
            initial = [max(0.01, 1-background), 0, 0, 1, 1, 0, background, 0, 0]
            lower = [0, -max_displacement, -max_displacement, *([width_bounds[0]]*2),
                     -onp.pi, -onp.inf, -onp.inf, -onp.inf]
            upper = [onp.inf, max_displacement, max_displacement, *([width_bounds[1]]*2),
                     onp.pi, onp.inf, onp.inf, onp.inf]
            initial[3:5] = [onp.clip(1., *width_bounds)] * 2
            fit = least_squares(lambda a: (_gaussian_plane(a, x, y)-normalized)[mask],
                                initial, bounds=(lower, upper), loss="soft_l1",
                                f_scale=0.05, max_nfev=2000)
            par = fit.x.copy()
            centre = seed + par[1:3]
            at_bound = bool(onp.linalg.norm(par[1:3]) >= max_displacement - 1e-3)
            success = bool(fit.success and par[0] > 1e-6)
            message = str(fit.message)
            model = _gaussian_plane(par, x, y) * scale
            par[[0, 6, 7, 8]] *= scale
        coordinates.append(onp.stack([u, v], axis=-1))
        patches.append(data)
        models.append(model)
        masks.append(mask)
        params.append(par)
        centres.append(centre)
        successes.append(success)
        bounds.append(at_bound)
        messages.append(message)
    return SplodgeFit(power.copy(), seeds.copy(), onp.array(centres),
                      onp.array(coordinates), onp.array(patches), onp.array(models),
                      onp.array(masks), onp.array(patches)-onp.array(models),
                      onp.array(params), onp.array(successes), onp.array(bounds),
                      tuple(messages), central_radius=central_radius)


def extract_phasors(
    frames, centres, *, offset=0.0, window=None, batch_size=256,
    noise_corner_fraction=0.25, noise_exclusion_radius=4.0,
) -> PhasorResult:
    """Direct centred Fourier sums and instantaneous complex noise power.

    frames has shape (N, ny, nx), centres (6, 2). Subtract scalar offset before
    applying window; None uses the RTC super-Gaussian exp(-radius**4), centred
    at (nx//2, ny//2). The Fourier phase uses this same fixed image origin.
    Phasors are unnormalized sums. noise_power has shape (N,) and estimates
    E[abs(complex noise)**2] in the same units from Fourier corners, assuming
    spatially white pixel noise. A direct Fourier kernel has unit noise gain
    relative to integer FFT bins, even for fractional centres.
    corner_fraction=0.25 includes |u| >= 0.375*nx and |v| >= 0.375*ny;
    exclude periodic disks around all fitted splodges and their conjugates.
    Nonfinite input frames return NaN measurements and valid_frames=False.
    No input array is modified, and no bad-pixel correction is applied.
    """
    frames = onp.asarray(frames)
    if frames.ndim != 3:
        raise ValueError("frames must have shape (frames, y, x)")
    sampler = _PhasorSampler(frames.shape[1:], centres, window,
                             noise_corner_fraction, noise_exclusion_radius)
    if not isinstance(batch_size, (int, onp.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not onp.isfinite(offset):
        raise ValueError("offset must be finite")
    outputs = [sampler.measure(frames[i:i+batch_size], offset)
               for i in range(0, len(frames), batch_size)]
    if not outputs:
        return PhasorResult(onp.empty((0, 6), complex), onp.empty(0), onp.empty(0, bool))
    return PhasorResult(*(onp.concatenate([getattr(o, name) for o in outputs])
                          for name in ("phasors", "noise_power", "valid_frames")))


def phasors_to_pd(phasors, noise_power, *, snr_threshold=4.5) -> PhaseDelayResult:
    """RTC wrap search and SNR-squared projection, with zero common piston.

    SNR = abs(phasor) / sqrt(E[abs(complex noise)**2]), not quadrature SNR.
    noise_power accepts a scalar, (N,), (N, 1), or (N, 6). Baseline phase is
    telescope_j - telescope_i. Only SNR > snr_threshold is used. Invalid noise
    or a disconnected four-telescope graph produces NaN pd_tel; baseline
    phases/SNRs remain available. Telescope phases have zero sum but can exceed
    +/-0.5 waves after the within-frame wrap search; there is no time unwrapping.
    """
    z = onp.asarray(phasors, dtype=complex)
    if z.ndim != 2 or z.shape[1] != 6:
        raise ValueError("phasors must have shape (frames, 6)")
    if not onp.isfinite(snr_threshold) or snr_threshold < 0:
        raise ValueError("snr_threshold must be finite and nonnegative")
    noise = onp.asarray(noise_power, dtype=float)
    if noise.ndim == 1:
        noise = noise[:, None]
    noise = onp.broadcast_to(noise, z.shape)
    noise_ok = onp.isfinite(noise) & (noise > 0)
    snr = onp.full(z.shape, onp.nan)
    onp.divide(onp.abs(z), onp.sqrt(onp.where(noise_ok, noise, onp.nan)), out=snr)
    baseline = onp.angle(z) / (2*onp.pi)
    usable = noise_ok & onp.isfinite(z) & onp.isfinite(snr) & (snr > snr_threshold)
    weights = onp.where(usable, snr, 0.)
    # Normalize each row before squaring to avoid overflow without changing projection.
    weights /= onp.maximum(weights.max(axis=1, keepdims=True), 1.)
    weights **= 2
    connected = _connected(usable)
    valid_noise = noise_ok.all(axis=1)
    valid = connected & valid_noise
    tel = onp.full((len(z), 4), onp.nan)
    for first in range(0, len(z), 2048):
        rows = onp.flatnonzero(valid[first:first+2048]) + first
        if len(rows) == 0:
            continue
        w = weights[rows]
        normal = onp.einsum("bi,nb,bj->nij", BASELINE_MATRIX, w, BASELINE_MATRIX)
        rhs = BASELINE_MATRIX.T[None, :, :] * w[:, None, :]
        projection = onp.linalg.solve(normal + onp.ones((4, 4))/4, rhs)
        x = onp.where(onp.isfinite(baseline[rows]), baseline[rows], 0.)
        direction = onp.where(x > 0, -1., 1.)
        best = onp.full(len(rows), onp.inf)
        best_tel = onp.zeros((len(rows), 4))
        # Same ascending mask order as RTC filter6: flip at most two baselines.
        for bits in range(64):
            if bits.bit_count() > 2:
                continue
            flip = ((bits >> onp.arange(6)) & 1)
            trial = x + direction * flip
            candidate = onp.einsum("nij,nj->ni", projection, trial)
            residual = trial - candidate @ BASELINE_MATRIX.T
            chi2 = onp.sum(w * residual**2, axis=1)
            # Equal physical minima can differ by integer telescope pistons.
            # Keep RTC mask order through floating-point ties, independent of batching.
            better = chi2 < best - 1e-12
            best[better] = chi2[better]
            best_tel[better] = candidate[better]
        tel[rows] = best_tel - best_tel.mean(axis=1, keepdims=True)
    return PhaseDelayResult(snr, baseline, tel, usable, valid_noise, connected, valid)


def extract_pd_from_fits(
    files: Sequence[str | Path], initial_centres, *, start=None, end=None,
    batch_size=256, snr_threshold=4.5, patch_size=7, central_radius=4.0,
    max_displacement=2.0, width_bounds=(0.3, 2.5), window=None,
    noise_corner_fraction=0.25, noise_exclusion_radius=4.0,
) -> ExtractionResult:
    """Two data passes with filename_time - frame_count/FPS + frame_index/FPS.

    Bounds are UTC and select [start, end). Supply all potentially intersecting
    paths, including the first completed cube filename after ``end``.
    GCDS only; offsets come from each file header. No dark or bad-pixel repair.
    """
    cubes = index_fits(files, start=start, end=end, timestamp="completed")
    if not cubes:
        raise ValueError("No FITS frames in requested interval")
    shape = cubes[0].shape[1:]
    win = _window(shape, window)
    for cube in cubes:
        if cube.shape[1:] != shape:
            raise ValueError("All cubes must have the same image shape")
        mode = next((cube.header[k] for k in
                     ("RO_MODE", "READMODE", "READOUT", "READOUT_MODE", "RDOUTMOD")
                     if k in cube.header), None)
        if str(mode).strip().upper() != "GCDS":
            raise ValueError(f"{cube.path}: GCDS required, got {mode!r}; "
                             "NDMR reset-validity handling is not implemented")
        if "OFFSET" not in cube.header or not onp.isfinite(float(cube.header["OFFSET"])):
            raise ValueError(f"{cube.path}: finite OFFSET header required")
    total = onp.zeros(shape, dtype=float)
    count = 0
    for cube in cubes:
        for _, frames in read_cube_batches(cube, batch_size):
            images, good = _prepare(frames, cube.header["OFFSET"], win)
            power = onp.abs(onp.fft.fft2(images[good], axes=(-2, -1)))**2
            # Accumulate by frame so batch size cannot change summation order.
            for frame_power in power:
                total += frame_power
            count += int(good.sum())
    if count == 0:
        raise ValueError("No finite frames available for centre fitting")
    fit = fit_splodge_centres(total/count, initial_centres, patch_size=patch_size,
                             central_radius=central_radius, max_displacement=max_displacement,
                             width_bounds=width_bounds)
    fit.frame_count = count
    if not onp.all(fit.success & ~fit.at_bound):
        raise SplodgeFitError(fit)
    sampler = _PhasorSampler(shape, fit.centres, win, noise_corner_fraction,
                             noise_exclusion_radius)
    chunks = []
    for file_index, cube in enumerate(cubes):
        for indices, frames in read_cube_batches(cube, batch_size):
            measured = sampler.measure(frames, cube.header["OFFSET"])
            chunks.append((cube.times(indices[0], indices[-1]+1),
                           onp.full(len(indices), file_index, dtype=int), indices,
                           onp.full(len(indices), cube.fps), measured))
    times, file_indices, frame_indices, fps = (
        onp.concatenate([chunk[i] for chunk in chunks]) for i in range(4))
    phasors, noise, valid_frames = (
        onp.concatenate([getattr(chunk[4], name) for chunk in chunks])
        for name in ("phasors", "noise_power", "valid_frames"))
    pd = phasors_to_pd(phasors, noise, snr_threshold=snr_threshold)
    return ExtractionResult(**vars(pd), times=times,
                            source_files=tuple(c.path for c in cubes),
                            file_indices=file_indices, frame_indices=frame_indices,
                            fps=fps, phasors=phasors, noise_power=noise,
                            valid_frames=valid_frames, fit=fit)


def plot_splodge_fits(fit: SplodgeFit):
    """Return (figure, axes) pairs for the full spectrum and six fit panels."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    ny, nx = fit.mean_power.shape
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    positive = fit.mean_power[fit.mean_power > 0]
    vmin = max(float(onp.percentile(positive, 10)), 1e-30) if positive.size else 1e-30
    vmax = max(float(positive.max()), vmin*10) if positive.size else 1e-29
    image = ax.imshow(onp.fft.fftshift(fit.mean_power), origin="lower",
                      extent=(-(nx//2)-.5, (nx-1)//2+.5, -(ny//2)-.5, (ny-1)//2+.5),
                      norm=LogNorm(vmin=vmin, vmax=vmax))
    for i, (seed, centre) in enumerate(zip(fit.initial_centres, fit.centres)):
        seed = (seed + onp.array([nx, ny])/2) % [nx, ny] - onp.array([nx, ny])/2
        centre = (centre + onp.array([nx, ny])/2) % [nx, ny] - onp.array([nx, ny])/2
        ax.plot(*seed, "+", color="cyan")
        if onp.isfinite(centre).all():
            ax.plot(*centre, "x", color="red")
        ax.annotate(BASELINES[i], centre if onp.isfinite(centre).all() else seed,
                    xytext=(5, 5), textcoords="offset points", color="red")
    ax.add_patch(plt.Circle((0, 0), fit.central_radius, fill=False, edgecolor="white", linestyle=":"))
    ax.set(xlabel="u (Fourier pixels)", ylabel="v (Fourier pixels)",
           title="Interval mean power: seeds (+), fitted centres (×)")
    fig.colorbar(image, ax=ax, label="Power (unnormalized FFT)")
    panels, axes = plt.subplots(6, 3, figsize=(11, 17), constrained_layout=True)
    for i in range(6):
        unmasked = fit.patches[i][fit.masks[i]]
        vmax = max(float(unmasked.max()), 1e-30) if unmasked.size else 1.
        for j, (values, title) in enumerate(zip(
                (fit.patches[i], fit.models[i], fit.residuals[i]), ("Data", "Model", "Residual"))):
            a = axes[i, j]
            data = onp.ma.array(values, mask=~fit.masks[i])
            finite = onp.abs(values[fit.masks[i] & onp.isfinite(values)])
            limit = max(float(finite.max()), 1e-30) if j == 2 and finite.size else vmax
            im = a.imshow(data, origin="lower", cmap="RdBu_r" if j == 2 else "viridis",
                          vmin=-limit if j == 2 else 0, vmax=limit)
            status = "OK" if fit.success[i] and not fit.at_bound[i] else "CORRECT SEED"
            a.set_title(f"{BASELINES[i]} {title} ({status})", fontsize=9)
            if j == 0:
                a.set_ylabel(f"u,v = {fit.centres[i,0]:.3f}, {fit.centres[i,1]:.3f}")
            panels.colorbar(im, ax=a, shrink=0.7)
    return (fig, ax), (panels, axes)


BASELINE_MATRIX = onp.array([[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1],
                [0, -1, 1, 0], [0, -1, 0, 1], [0, 0, -1, 1]], dtype=float)


def _centres(centres):
    centres = onp.asarray(centres, dtype=float)
    if centres.shape != (6, 2) or not onp.all(onp.isfinite(centres)):
        raise ValueError("centres must contain six finite signed (u, v) pairs")
    return centres


def _gaussian_plane(a, x, y):
    amp, cx, cy, sx, sy, theta, b, bx, by = a
    dx, dy = x-cx, y-cy
    xx = onp.cos(theta)*dx + onp.sin(theta)*dy
    yy = -onp.sin(theta)*dx + onp.cos(theta)*dy
    return amp*onp.exp(-0.5*((xx/sx)**2+(yy/sy)**2)) + b + bx*x + by*y


def _window(shape, window):
    ny, nx = shape
    if window is None:
        y, x = onp.indices(shape)
        radius = ((x-nx//2)/(nx//2))**2 + ((y-ny//2)/(ny//2))**2
        return onp.exp(-radius**2)
    window = onp.asarray(window, dtype=float)
    if window.shape != tuple(shape) or not onp.isfinite(window).all() or not onp.any(window):
        raise ValueError("window must be finite, nonzero, and match the image shape")
    return window


def _prepare(frames, offset, window):
    images = onp.asarray(frames, dtype=float) - float(offset)
    good = onp.isfinite(images).all(axis=(-2, -1))
    images[~good] = 0
    return images * window, good


class _PhasorSampler:
    def __init__(self, shape, centres, window, corner_fraction, exclusion_radius):
        self.window = _window(shape, window)
        centres = _centres(centres)
        if not 0 < corner_fraction < 0.5 or exclusion_radius < 0:
            raise ValueError("Invalid noise corner fraction or exclusion radius")
        ny, nx = shape
        y, x = onp.indices(shape)
        # A fixed physical origin, also at fractional frequencies: no bin interpolation.
        phase = (centres[:, 0, None, None]*(x-nx//2)/nx +
                 centres[:, 1, None, None]*(y-ny//2)/ny)
        self.kernels = onp.exp(-2j*onp.pi*phase).reshape(6, -1).T.copy()
        u = onp.fft.fftfreq(nx)*nx
        v = onp.fft.fftfreq(ny)*ny
        u, v = onp.meshgrid(u, v)
        self.noise_mask = ((onp.abs(u) >= nx*(0.5-corner_fraction/2)) &
                           (onp.abs(v) >= ny*(0.5-corner_fraction/2)))
        for centre in centres:
            for sign in (-1, 1):
                du = (u-sign*centre[0]+nx/2) % nx - nx/2
                dv = (v-sign*centre[1]+ny/2) % ny - ny/2
                self.noise_mask &= onp.hypot(du, dv) > exclusion_radius
        if self.noise_mask.sum() < 4:
            raise ValueError("Too few noise corner bins after splodge exclusion")

    def measure(self, frames, offset):
        images, good = _prepare(frames, offset, self.window)
        phasors = images.reshape(len(images), -1) @ self.kernels
        fourier = onp.fft.fft2(images, axes=(-2, -1))
        noise = onp.mean(onp.abs(fourier[:, self.noise_mask])**2, axis=1)
        phasors[~good] = onp.nan
        noise[~good] = onp.nan
        return PhasorResult(phasors, noise, good)


def _connected(usable):
    # Connectivity is a graph property; do not confuse a tiny SNR weight with a missing edge.
    patterns = onp.sum(usable.astype(int) * (1 << onp.arange(6)), axis=1)
    lookup = onp.zeros(64, dtype=bool)
    for bits in range(64):
        edges = ((bits >> onp.arange(6)) & 1).astype(bool)
        lookup[bits] = onp.linalg.matrix_rank(BASELINE_MATRIX[edges]) == 3 if edges.any() else False
    return lookup[patterns]


@dataclass
class TelescopePDSpectrum:
    frequencies: onp.ndarray
    power: onp.ndarray
    fps: float
    segment_count: int
    run_count: int
    sample_count: int


def telescope_pd_welch(result: ExtractionResult, *, nperseg=1024) -> TelescopePDSpectrum:
    """Average Welch estimates of the supplied telescope PD within valid cube runs.

    ExtractionResult.pd_tel is wrapped by default; this function applies no
    temporal unwrapping. It also accepts a result with pd_tel replaced by an
    explicitly unwrapped array.

    Uses native header FPS, a Hann window, 50% overlap, and constant detrending.
    Each run stays inside one cube, with contiguous source frame indices and
    uniform estimated UTC sampling (2 ns rounding tolerance). Runs shorter than
    nperseg are omitted. Estimates are weighted by their Welch segment counts.
    Cubes with different native rates require separate calls.
    """
    from scipy.signal import welch

    if not isinstance(nperseg, (int, onp.integer)) or nperseg < 2:
        raise ValueError("nperseg must be an integer >= 2")
    rates = onp.unique(result.fps)
    if len(rates) != 1:
        raise ValueError("Select cubes with one native FPS for each spectrum")
    fps = float(rates[0])
    total = onp.zeros((nperseg//2+1, 4))
    segments, runs, samples = 0, 0, 0
    for first, stop in valid_frame_runs(result):
        length = stop-first
        if length < nperseg:
            continue
        count = 1 + (length-nperseg)//(nperseg-nperseg//2)
        frequencies, power = welch(result.pd_tel[first:stop], fs=fps,
                                   nperseg=nperseg, noverlap=nperseg//2,
                                   window="hann", detrend="constant", axis=0)
        total += count*power
        segments += count
        runs += 1
        samples += length
    if not segments:
        raise ValueError("No valid uniform within-cube runs long enough for nperseg")
    return TelescopePDSpectrum(frequencies, total/segments, fps, segments, runs, samples)



def valid_frame_runs(result: ExtractionResult):
    """Yield [first, stop) valid uniform runs, split at every cube/frame/time gap."""
    good = result.valid & onp.isfinite(result.pd_tel).all(axis=1)
    if not len(good):
        return
    delta_ns = onp.diff(result.times).astype('timedelta64[ns]').astype(onp.int64)
    contiguous = ((onp.diff(result.fps) == 0) &
                  (onp.diff(result.file_indices) == 0) &
                  (onp.diff(result.frame_indices) == 1) &
                  (onp.abs(delta_ns-1e9/result.fps[:-1]) <= 2))
    starts = onp.flatnonzero(good & ~onp.r_[False, good[:-1] & contiguous])
    stops = onp.flatnonzero(good & ~onp.r_[good[1:] & contiguous, False]) + 1
    yield from zip(starts, stops)
