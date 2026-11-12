# heimdallr-dr

## Offline K1 phase-delay extraction

`heimdallrdr.fringe` runs on an offline analysis machine, without RTC services.
It uses existing NumPy, SciPy, Astropy and Matplotlib dependencies.

```python
from heimdallrdr.fringe import extract_pd_from_fits, plot_splodge_fits

result = extract_pd_from_fits(
    files,                    # explicit K1 cube paths, including the next completion
    initial_centres,          # six signed (u, v) Fourier-pixel coordinates
    start="2026-04-03T00:59:30Z",
    end="2026-04-03T01:01:43Z",
    batch_size=256,
    snr_threshold=4.5,
)
(full_figure, full_axis), (patch_figure, patch_axes) = plot_splodge_fits(result.fit)
```

All baseline arrays use **12, 13, 14, 23, 24, 34** order. Positive baseline phase
is telescope *j* minus telescope *i*. Coordinates are signed `(u, v)` in an
**unshifted** full Fourier spectrum, while images use `(y, x)` indexing.
Signed coordinates select the baseline's conjugate automatically; do not add
another RTC sign correction. Telescope and baseline PD are in K1 wavelengths.

### Public array and fitting interfaces

- `fit_splodge_centres(mean_power, initial_centres, *, patch_size=7,
  central_radius=4, max_displacement=2, width_bounds=(0.3, 2.5))` returns
  `SplodgeFit`. It fits an elliptical Gaussian and planar background with
  robust soft-L1 least squares. Central-radius pixels are excluded, and patch
  indexing is periodic across Fourier boundaries. Widths are Gaussian sigmas.
  Fits at a centre displacement of two pixels or more need corrected seeds.
- `extract_phasors(frames, centres, *, offset=0, window=None, batch_size=256,
  noise_corner_fraction=0.25, noise_exclusion_radius=4)` returns `PhasorResult`
  with `(N, 6)` complex phasors, `(N,)` noise powers, and finite-frame flags.
  It evaluates precomputed complex Fourier kernels at the exact fractional
  coordinates with origin `(nx//2, ny//2)`, rather than interpolating FFT bins.
- `phasors_to_pd(phasors, noise_power, *, snr_threshold=4.5)` returns
  `PhaseDelayResult`. Noise may be scalar, `(N,)`, `(N, 1)` or `(N, 6)`.
  It uses strictly above-threshold SNR-squared weights, the RTC's ascending
  wrap search (at most two baseline flips), and weighted telescope projection.
  Equal-cost wrap choices retain the first mask within floating-point tolerance
  to make processing independent of batch size.
- `plot_splodge_fits(fit)` returns the two `(figure, axes)` pairs above, including
  failed fits. It does not show or save figures. Diagnostics retain mean power,
  seeds, centres, patch coordinates/data/models/masks/residuals, model parameters,
  optimizer messages, success flags and centre-bound flags.

`pd_snr = abs(phasor) / sqrt(noise_power)` is a **complex-amplitude SNR**:
noise power means `E[abs(complex noise)**2]`, rather than the variance of one
quadrature. The default window is the RTC super-Gaussian. Noise is the mean
instantaneous power in high-frequency corners (`|u| >= 0.375*nx`,
`|v| >= 0.375*ny`), excluding radius-four periodic disks around fitted centres
and their conjugates. For spatially white detector noise the direct kernels
and integer FFT bins have the same complex noise power. Correlated detector
noise or residual signals in the corners can bias this estimate.

### FITS selection and extraction result

There are two streaming pixel passes: average the power of all finite selected
frames, fit six fixed centres, then extract all selected phasors. Primary-image
and compressed-image cubes share lazy FITS section reads with `DataBundler`.
Each file must provide positive FPS, GCDS readout mode (`RO_MODE` or the legacy
aliases), and finite `OFFSET`. Subtract each file's offset before applying the
window. There is no additional dark subtraction or automatic bad-pixel repair.
NDMR raises an explicit error until reset-validity handling is implemented.
The supplied data must be K1; GD and historical RTC emulation are not provided.

The extractor explicitly interprets the filename as a **completed-cube** time:

```
estimated UTC = filename_time - frame_count/FPS + frame_index/FPS
```

Supply all potentially intersecting files, **including the first completion
filename after the requested end**. Bounds select `[start, end)` exactly.
Dates come from `UTDATE` or a `YYYYMMDD` parent directory; aware bounds convert
to UTC and naive bounds denote UTC. Returned times are `datetime64[ns]` UTC.
They are estimates based on header FPS, not hardware exposure timestamps.
`DataBundler` keeps its existing filename-as-start timestamp default and loader
return signatures.

`ExtractionResult` contains:

| Field | Shape / meaning |
| --- | --- |
| `times` | `(N,)` estimated UTC frame times |
| `source_files` | Ordered tuple of contributing paths |
| `file_indices`, `frame_indices` | `(N,)` indices into `source_files` and the original cube |
| `fps` | `(N,)` native per-frame rate from the contributing header |
| `phasors`, `pd_snr`, `pd_baseline` | `(N, 6)` complex measurements, SNR and wrapped baseline PD |
| `noise_power` | `(N,)` instantaneous complex noise power |
| `pd_tel` | `(N, 4)` telescope PD with zero common piston |
| `valid_baselines` | `(N, 6)` finite measurements above SNR threshold |
| `valid_frames`, `valid_noise`, `connected`, `valid` | `(N,)` finite input, valid noise, connected graph, usable telescope PD |
| `fit` | `SplodgeFit` diagnostics, including averaged frame count |

Disconnected baseline graphs or invalid noise estimates give NaN telescope PD;
measured baseline outputs remain available. Failed/bounded centre fits raise
`SplodgeFitError`, whose `.fit` retains diagnostics for correcting seeds.
There is **no temporal unwrapping in the extractor**. The within-frame wrap
search can produce telescope values outside ±0.5 wavelengths and integer-piston
choices can change between frames.

### Spectra and the April example

`valid_frame_runs(result)` yields contiguous valid runs within individual cubes,
checking source frame indices and native time spacing. `telescope_pd_welch`
averages Hann-window Welch estimates over runs long enough for `nperseg`, using
header FPS, 50% overlap, constant detrending and segment-count weighting. It
never bridges missing data or infers frame rate from cube save-time jitter.
Different native rates require separate spectra. It processes `pd_tel` as
supplied and does not unwrap it.

The workspace example
[`asg-analysis/ddspc/x_analyse_prev_UT_pspec.py`](../asg-analysis/ddspc/x_analyse_prev_UT_pspec.py)
keeps its settings-availability and telemetry analysis. Its new extraction
uses explicit geometry-derived seeds and the April interval above. Environment
settings `HEIMDALLR_FITS_ROOT`, `HEIMDALLR_PD_START`, `HEIMDALLR_PD_END`,
`HEIMDALLR_PD_BATCH_SIZE`, and `HEIMDALLR_PD_THRESHOLD`
configure it; `HEIMDALLR_EXTRACT_FITS=0` skips the new section.

Before plotting telescope PD, the example **attempts temporal unwrapping** of
projected baseline differences, then projects back into zero-common-piston
telescope space. This avoids fractional common-piston jumps from unwrapping
telescopes independently. The example assumes the selected cubes form one
continuous stream with no acquisition gaps or overlaps, so unwrapping continues
across cube boundaries. Invalid telescope-PD samples are linearly interpolated
at their original frame indices after unwrapping the valid measurements;
leading/trailing invalid samples use the nearest valid value. True baseline
changes above half a wave between valid measurements are ambiguous;
this heuristic is not evidence of the absolute fringe order. The example saves
fit diagnostics, raw baseline PD, SNR, attempted unwrapped telescope traces,
and attempted unwrapped PSDs with a linear frequency axis using its `save_plot` helper.
Time plots use uniform native-FPS spacing anchored to the first selected frame.
Orange dashed lines mark cube starts on that same continuous clock in the
unwrapped telescope-PD plot.
The final spectra use Matplotlib `Axes.psd` on the complete interpolated series,
with `NFFT=1024`, native `Fs`, a Hann window, 512-sample overlap and mean
detrending. Matplotlib plots PSD in dB; the frequency axis stays linear.
Raw extraction results and telemetry distribution comparisons stay wrapped.

Run offline checks from `heimdallr-dr`:

```bash
conda run -n asg python -m unittest discover -s tests -q
```

### Offline validation: 3 April 2026, 00:59:30–01:01:43 UTC

Validated in the `asg` environment using
`/media/taras/T7/work/asgard/Apr2026-comp`. The example completed its original
settings and telemetry work and saved all new figures. The extraction selected
**132,983 frames in 28 cubes**, with **132,970 valid telescope-PD frames
(99.990% of selected)**. Nominal saved exposure is 132.983 seconds out of the
133-second request; valid nominal exposure is 132.970 seconds. These sums do
not imply that independently saved cubes have perfectly contiguous timestamps.
Under the filename-derived timing model, merging per-frame exposure intervals
gives 132.904 seconds of selected coverage and 132.891 seconds of valid coverage
(99.918% of the requested interval). These are model-based estimates, not
measurements of acquisition gaps or duplicate exposures. In the current saver
implementation, the filename timestamp is taken when the save thread wakes;
variable scheduling latency can shift adjacent cube timestamps even when
acquisition is continuous. Across the 27 boundaries, completion spacings differ
from five seconds by -44 to +50 ms. Comparing the last/first 100 images at every
boundary found no exact repeated images. The headers have no per-frame hardware
timestamps or frame counters establishing whether frames were lost.
All 132,983 noise estimates are valid;
the 13 invalid telescope frames have disconnected above-threshold graphs.
All SNRs were finite; the largest valid absolute telescope-phase sum was
**1.80 × 10⁻¹⁶ K1 wavelengths**.

| Baseline | Fitted u (pixels) | Fitted v (pixels) |
| --- | ---: | ---: |
| 12 | 0.163025 | -11.176220 |
| 13 | 3.303797 | -5.962940 |
| 14 | 10.730508 | -6.493416 |
| 23 | 3.235743 | 5.155610 |
| 24 | 10.677103 | 4.762200 |
| 34 | 7.380577 | -0.414591 |

All six full-interval data/model/residual panels were visually inspected.
Each fit locates a compact peak and passes the optimizer/displacement checks;
none reaches the centre-search bound. Fitted Gaussian sigmas range from 0.873
to 0.980 pixels. Residual neighboring-peak wings remain,
particularly in the baseline-13 patch. The central mask excludes the strong
central contribution in the baseline-23 patch. Diagnostics are available in
[`splodge_fits_20260403T005930_010143.png`](../asg-analysis/ddspc/plots/x_analyse_prev_UT_pspec/splodge_fits_20260403T005930_010143.png).

The final attempted unwrapped PSDs treat **132,983 frames as one continuous
series at 1,000 Hz**, with the **13 invalid PD samples interpolated** at their
original positions. Matplotlib `Axes.psd` uses `NFFT=1024` and 512-sample overlap
(258 complete windows), allowing windows to cross cube boundaries. The
continuous unwrapped traces and PSDs were regenerated and visually inspected.
Unwrapping reveals narrow peaks suppressed by wrapping; it does not establish
an absolute fringe order. The plotting series is finite throughout, with
maximum absolute common piston 8.88 × 10⁻¹⁶ K1 wavelengths. A second full extraction with batches of
1,024 frames reproduces the 256-frame-batch counts and reported centres.

There are **1,328 telemetry entries** in the same interval. Median complex
amplitude SNRs provide context:

| Baseline | Offline median SNR | Telemetry median SNR |
| --- | ---: | ---: |
| 12 | 12.9980 | 10.6450 |
| 13 | 22.0290 | 15.3950 |
| 14 | 17.7083 | 16.6350 |
| 23 | 40.5109 | 40.0200 |
| 24 | 25.9929 | 25.3800 |
| 34 | 32.6880 | 28.2150 |

The raw telescope-PD 5th/50th/95th percentiles (K1 wavelengths) are:

| Telescope | Offline | Telemetry |
| --- | --- | --- |
| 1 | -0.3866 / -0.0029 / 0.3855 | -0.3519 / 0.0025 / 0.3725 |
| 2 | -0.3880 / -0.0038 / 0.3796 | -0.3640 / 0.0100 / 0.3900 |
| 3 | -0.3806 / 0.0078 / 0.3897 | -0.3786 / -0.0060 / 0.3732 |
| 4 | -0.2096 / -0.0009 / 0.2094 | -0.3776 / -0.0110 / 0.3800 |

The example prints all SNR and raw-PD percentile comparisons. Different
sampling, preprocessing, estimator history and equal-cost integer-piston
choices can change these distributions; numerical equality is not an
acceptance criterion. No instrument hardware or live services were used.
The 20 package tests and three example-unwrapping tests pass, covering centre
recovery, phases/signs, periodic boundaries, noise scaling, wrap projection,
invalid/disconnected data, FITS layouts/time selection/batch agreement, native
Welch runs, continuous plotting-only unwrapping, and interpolation of invalid PD.
