import numpy as np
from astropy.io import fits
import pathlib
import datetime
import warnings
import bisect
import tqdm.auto as tqdm


class DataBundler:
    def __init__(self, base_path):
        """Create a data bundler rooted at a base directory.

        Parameters
        ----------
        base_path : str or pathlib.Path
            Either a root directory containing YYYYMMDD folders or a single
            YYYYMMDD folder.
        """
        self.base_path = pathlib.Path(base_path)

    def _coerce_utc_naive(self, value):
        """Convert datetimes to naive UTC for internal comparisons."""
        if value.tzinfo is None:
            return value
        return value.astimezone(datetime.timezone.utc).replace(tzinfo=None)

    def _extract_camera_settings(self, header):
        """Extract camera settings used to check consistency across files.

        Parameters
        ----------
        header : astropy.io.fits.Header
            FITS header to inspect.

        Returns
        -------
        dict
            Mapping with keys ``fps``, ``readout_mode``, and ``camera_gain``.
            Values are ``None`` when no matching header keyword is present.
        """

        def _first_present(keys):
            for key in keys:
                if key in header:
                    return header[key]
            return None

        return {
            "fps": _first_present(("FPS", "FRAMERATE", "FRAME_RATE")),
            "readout_mode": _first_present(
                ("READMODE", "READOUT", "READOUT_MODE", "RDOUTMOD")
            ),
            "camera_gain": _first_present(("GAIN", "CAMGAIN", "CAM_GAIN")),
        }

    def _warn_if_camera_settings_changed(self, settings_by_file):
        """Warn if fps/readout mode/gain change across selected files.

        Parameters
        ----------
        settings_by_file : list[tuple[datetime.datetime, dict]]
            Per-file camera settings keyed by cube start time.
        """
        if not settings_by_file:
            return

        unique_values = {
            "fps": set(),
            "readout_mode": set(),
            "camera_gain": set(),
        }
        for _, settings in settings_by_file:
            for key in unique_values:
                unique_values[key].add(settings.get(key))

        changed = {
            k: sorted(v, key=lambda x: str(x))
            for k, v in unique_values.items()
            if len(v) > 1
        }
        if not changed:
            return

        first_file = settings_by_file[0][0]
        last_file = settings_by_file[-1][0]
        warnings.warn(
            "Camera settings changed within requested time range "
            f"({first_file} -> {last_file}): {changed}",
            RuntimeWarning,
            stacklevel=2,
        )

    def _cube_overlap_slice(
        self,
        cube_start: datetime.datetime,
        n_frames: int,
        fps: float,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ):
        """Compute frame-index overlap of a cube with a requested time window.

        Parameters
        ----------
        cube_start : datetime.datetime
            Timestamp of the first frame in the cube.
        n_frames : int
            Number of frames in the cube.
        fps : float
            Frame rate in frames per second.
        start_time : datetime.datetime
            Requested start time.
        end_time : datetime.datetime
            Requested end time.

        Returns
        -------
        tuple[int, int] or None
            Frame slice ``(start_idx, end_idx)`` with ``end_idx`` exclusive,
            or ``None`` when there is no overlap.
        """
        cube_duration = datetime.timedelta(seconds=n_frames / fps)
        cube_end = cube_start + cube_duration

        overlap_start = max(start_time, cube_start)
        overlap_end = min(end_time, cube_end)
        if overlap_start >= overlap_end:
            return None

        start_offset_s = (overlap_start - cube_start).total_seconds()
        end_offset_s = (overlap_end - cube_start).total_seconds()

        start_idx = int(np.floor(start_offset_s * fps))
        end_idx = int(np.ceil(end_offset_s * fps))

        start_idx = max(0, min(start_idx, n_frames))
        end_idx = max(0, min(end_idx, n_frames))
        if end_idx <= start_idx:
            return None

        return start_idx, end_idx

    def _iter_relevant_date_dirs(self, root, start_time, end_time):
        """Yield existing YYYYMMDD folders relevant to a time window.

        Includes one day prior to ``start_time`` to catch cubes that start
        before midnight but overlap the requested interval.
        """
        day = start_time.date() - datetime.timedelta(days=1)
        end_day = end_time.date()
        while day <= end_day:
            day_dir = root / day.strftime("%Y%m%d")
            if day_dir.is_dir():
                yield day, day_dir
            day += datetime.timedelta(days=1)

    def _validate_time_range(self, start_time, end_time):
        """Validate that ``start_time`` and ``end_time`` define a non-empty range."""
        start_time = self._coerce_utc_naive(start_time)
        end_time = self._coerce_utc_naive(end_time)
        if end_time <= start_time:
            raise ValueError("end_time must be after start_time.")

    def _resolve_root(self):
        """Resolve the search root from ``self.base_path``."""
        pth = self.base_path
        if pth.name.isdigit() and len(pth.name) == 8:
            return pth.parent
        return pth

    def _iter_candidate_files(self, crop, start_time, end_time):
        """Yield candidate files likely to overlap the requested range.

        Parameters
        ----------
        crop : str
            Camera/crop prefix in filenames.
        start_time : datetime.datetime
            Requested start time.
        end_time : datetime.datetime
            Requested end time.

        Yields
        ------
        tuple[datetime.datetime, pathlib.Path]
            Parsed file start time and file path.

        Notes
        -----
        This narrows work by only yielding files whose starts are inside the
        requested interval plus one predecessor file per day.
        """
        root = self._resolve_root()
        pth = self.base_path
        start_time = self._coerce_utc_naive(start_time)
        end_time = self._coerce_utc_naive(end_time)

        relevant_dirs = list(self._iter_relevant_date_dirs(root, start_time, end_time))
        if not relevant_dirs and pth.is_dir():
            relevant_dirs = [(start_time.date(), pth)]

        for day, date_dir in relevant_dirs:
            file_entries = []
            for file in sorted(date_dir.glob(f"{crop}_T*.fits")):
                try:
                    time_str = file.stem.split("_T", 1)[1]
                    time_of_day = datetime.datetime.strptime(
                        time_str, "%H_%M_%S.%f"
                    ).time()
                except (IndexError, ValueError):
                    continue

                file_time = datetime.datetime.combine(day, time_of_day)
                file_entries.append((file_time, file))

            if not file_entries:
                continue

            file_times = [t for t, _ in file_entries]
            left = max(0, bisect.bisect_left(file_times, start_time) - 1)
            right = bisect.bisect_right(file_times, end_time)

            for file_time, file in file_entries[left:right]:
                yield file_time, file

    def _frame_times_for_slice(self, file_time, start_idx, end_idx, fps):
        """Build per-frame timestamps for a slice in ``datetime64[us]`` format."""
        start_us = np.datetime64(file_time, "us")
        frame_idx = np.arange(start_idx, end_idx, dtype=np.float64)
        dt_us = np.rint((frame_idx / fps) * 1e6).astype(np.int64)
        return start_us + dt_us.astype("timedelta64[us]")

    def _build_overlap_plan(self, crop, start_time, end_time):
        """Build an overlap plan shared by loading and reduction methods.

        Returns
        -------
        overlap_plan : list[tuple]
            Entries ``(file_time, file, start_idx, end_idx, fps)`` sorted by
            ``file_time``.
        settings_by_file : list[tuple[datetime.datetime, dict]]
            Camera settings for files that contribute overlap, keyed by cube
            start time.
        """
        overlap_plan = []
        settings_by_file = []

        start_time = self._coerce_utc_naive(start_time)
        end_time = self._coerce_utc_naive(end_time)

        for file_time, file in self._iter_candidate_files(crop, start_time, end_time):
            try:
                with fits.open(file) as data:
                    hdu = data[1]  # type: ignore[index]
                    header = hdu.header  # type: ignore[attr-defined]
                    cube = hdu.data  # type: ignore[attr-defined]

                    if cube is None:
                        continue

                    settings = self._extract_camera_settings(header)
                    fps_value = settings.get("fps")
                    try:
                        fps = float(fps_value) if fps_value is not None else None
                    except (TypeError, ValueError):
                        fps = None

                    if fps is None or fps <= 0:
                        raise ValueError(
                            f"Invalid or missing FPS in {file}: {fps_value}. "
                            "Cannot compute partial cube overlap."
                        )

                    overlap = self._cube_overlap_slice(
                        cube_start=file_time,
                        n_frames=cube.shape[0],
                        fps=fps,
                        start_time=start_time,
                        end_time=end_time,
                    )
                    if overlap is None:
                        continue

                    start_idx, end_idx = overlap
                    settings_by_file.append((file_time, settings))
                    overlap_plan.append((file_time, file, start_idx, end_idx, fps))
            except Exception as e:
                print(f"Error loading {file}: {e}")

        overlap_plan.sort(key=lambda x: x[0])
        return overlap_plan, settings_by_file

    def load_between(
        self, crop: str, start_time: datetime.datetime, end_time: datetime.datetime
    ):
        """Load data between two datetimes.

        TODO: check what % of the requested time range is actually covered by returned frames
        and provide this as feedback to the user.

        Returns
        -------
        times_used : list[datetime.datetime]
            Start times of cubes that contributed frames.
        frames : np.ndarray
            Concatenated frame stack of shape (n_frames, ny, nx).
        frame_times : np.ndarray
            Per-frame timestamps as ``datetime64[us]`` aligned 1:1 with
            ``frames``.
        """
        self._validate_time_range(start_time, end_time)

        start_time = self._coerce_utc_naive(start_time)
        end_time = self._coerce_utc_naive(end_time)

        overlap_plan, settings_by_file = self._build_overlap_plan(
            crop, start_time, end_time
        )

        if not overlap_plan:
            raise ValueError("No files found in requested range.")

        selected = []
        times_used = []

        for file_time, file, start_idx, end_idx, fps in tqdm.tqdm(
            overlap_plan, desc="Loading frames"
        ):
            try:
                with fits.open(file) as data:
                    hdu = data[1]  # type: ignore[index]
                    cube = hdu.data  # type: ignore[attr-defined]
                    if cube is None:
                        continue

                    frame_slice = cube[start_idx:end_idx]
                    if frame_slice.shape[0] == 0:
                        continue

                    frame_times_slice = self._frame_times_for_slice(
                        file_time, start_idx, end_idx, fps
                    )
                    times_used.append(file_time)
                    selected.append((file_time, frame_slice, frame_times_slice))
            except Exception as e:
                print(f"Error loading {file}: {e}")

        if not selected:
            raise ValueError("No frames found in requested range.")

        self._warn_if_camera_settings_changed(settings_by_file)

        selected.sort(key=lambda x: x[0])
        times_used.sort()
        frames = np.concatenate([frame for _, frame, _ in selected])
        frame_times = np.concatenate([times for _, _, times in selected])
        return times_used, frame_times, frames

    def apply_fn_to_frames(
        self, crop: str, start_time: datetime.datetime, end_time: datetime.datetime, fn
    ):
        """Apply a per-frame function over a time range without storing full frames.

        For 30GB RAM and Heimdallr/Baldr crop sizes this is needed for processing
        longer than 1hr at 1000Hz (otherwise you can fit the images).

        # TODO: make this also accept time dependent functions fn(frame, time)

        This performs two passes:
        1) Scan relevant files, validate headers, and build an overlap plan.
        2) Re-read planned slices, apply ``fn(img)`` to each frame, store only outputs.

        Returns
        -------
        times_used : list[datetime.datetime]
            Start times of cubes that contributed processed outputs.
        frame_times : np.ndarray
            Per-frame timestamps as ``datetime64[us]`` aligned 1:1 with results.
        results : np.ndarray
            Concatenated function outputs. Shape is ``(n_frames, ...)``.
        """
        self._validate_time_range(start_time, end_time)

        start_time = self._coerce_utc_naive(start_time)
        end_time = self._coerce_utc_naive(end_time)

        overlap_plan, settings_by_file = self._build_overlap_plan(
            crop, start_time, end_time
        )
        if not overlap_plan:
            raise ValueError("No files found in requested range.")

        self._warn_if_camera_settings_changed(settings_by_file)
        overlap_plan.sort(key=lambda x: x[0])

        # Pass 2: apply fn per frame and keep only computed outputs.
        times_used = []
        frame_times_chunks = []
        results_chunks = []

        for file_time, file, start_idx, end_idx, fps in tqdm.tqdm(
            overlap_plan, desc="Processing frames"
        ):
            try:
                with fits.open(file) as data:
                    hdu = data[1]  # type: ignore[index]
                    cube = hdu.data  # type: ignore[attr-defined]
                    if cube is None:
                        continue

                    frame_slice = cube[start_idx:end_idx]
                    n_frames = frame_slice.shape[0]
                    if n_frames == 0:
                        continue

                    frame_times_chunk = self._frame_times_for_slice(
                        file_time, start_idx, end_idx, fps
                    )

                    first_out = fn(frame_slice[0])
                    first_arr = np.asarray(first_out)
                    out_shape = (n_frames,) + first_arr.shape
                    out_chunk = np.empty(out_shape, dtype=first_arr.dtype)
                    out_chunk[0] = first_arr
                    for i in range(1, n_frames):
                        out_chunk[i] = np.asarray(fn(frame_slice[i]))

                    times_used.append(file_time)
                    frame_times_chunks.append(frame_times_chunk)
                    results_chunks.append(out_chunk)
            except Exception as e:
                print(f"Error applying function to {file}: {e}")

        if not results_chunks:
            raise ValueError("No frame results produced in requested range.")

        frame_times = np.concatenate(frame_times_chunks)
        results = np.concatenate(results_chunks)
        return times_used, frame_times, results

    def load_camera_settings(
        self, start_time: datetime.datetime, end_time: datetime.datetime
    ):
        """Load camera settings for files overlapping a time range.

        Returns
        -------
        settings_by_file : list[tuple[datetime.datetime, dict]]
            Camera settings for files that overlap the requested range,
            keyed by cube start time.
        """
        start_time = self._coerce_utc_naive(start_time)
        end_time = self._coerce_utc_naive(end_time)

        _, settings_by_file = self._build_overlap_plan(
            crop="baldr1",  # note that crop is irrelevant here since all crops share the same camera
            start_time=start_time,
            end_time=end_time,
        )
        return settings_by_file
