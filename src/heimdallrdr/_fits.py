"""Shared cube metadata, time indexing and bounded image reads."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re

from astropy.io import fits
import numpy as onp


def utc64(value):
    """Naive values denote UTC; aware datetimes are converted to UTC."""
    if isinstance(value, str):
        value = value.strip()
        offset = re.search(r"([+-])(\d{2}):?(\d{2})$", value)
        if offset:
            hours, minutes = int(offset[2]), int(offset[3])
            if hours >= 24 or minutes >= 60:
                raise ValueError("Invalid UTC offset")
            sign = 1 if offset[1] == "+" else -1
            delta = onp.timedelta64(sign * (hours * 60 + minutes), "m")
            return onp.datetime64(value[:offset.start()], "ns") - delta
        return onp.datetime64(value.removesuffix("Z"), "ns")
    if isinstance(value, datetime) and value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return onp.datetime64(value, "ns")


@dataclass(frozen=True)
class CubeSelection:
    path: Path
    hdu_index: int
    shape: tuple
    header: fits.Header
    filename_time: onp.datetime64
    cube_start: onp.datetime64
    fps: float
    first: int
    stop: int

    def times(self, first=None, stop=None):
        indices = onp.arange(self.first if first is None else first,
                             self.stop if stop is None else stop)
        delta = onp.rint(indices / self.fps * 1e9).astype("timedelta64[ns]")
        return self.cube_start + delta


def inspect_cube(path, *, filename_time=None, timestamp="start", start=None, end=None):
    """Index a primary or compressed cube without decompressing its pixels.

    ``start`` preserves DataBundler's filename timestamp convention;
    ``completed`` places the first frame N/FPS before the filename timestamp.
    The exact rounded timestamps returned by ``times`` select [start, end).
    """
    path = Path(path)
    if timestamp not in ("start", "completed"):
        raise ValueError("timestamp must be 'start' or 'completed'")
    with fits.open(path) as hdus:
        candidates = [(i, h) for i, h in enumerate(hdus)
                      if isinstance(h, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU))
                      and len(h.shape) == 3]
        if len(candidates) != 1:
            raise ValueError(f"{path}: expected exactly one 3D image cube")
        hdu_index, hdu = candidates[0]
        header = hdus[0].header.copy()
        header.update(hdu.header)
        shape = hdu.shape
    fps = next((header[k] for k in ("FPS", "FRAMERATE", "FRAME_RATE") if k in header), None)
    if fps is None or not onp.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError(f"{path}: invalid or missing FPS: {fps}")
    fps = float(fps)
    if filename_time is None:
        match = re.search(r"_T(\d{2}_\d{2}_\d{2}(?:\.\d+)?)\.fits", path.name)
        if match is None:
            raise ValueError(f"{path}: expected a _THH_MM_SS.sss.fits filename")
        date = header.get("UTDATE")
        if date is None and re.fullmatch(r"\d{8}", path.parent.name):
            date = datetime.strptime(path.parent.name, "%Y%m%d").date().isoformat()
        if date is None:
            raise ValueError(f"{path}: missing UTDATE and YYYYMMDD parent directory")
        filename_time = f"{str(date)[:10]}T{match[1].replace('_', ':')}"
    filename_time = utc64(filename_time)
    cube_start = filename_time
    if timestamp == "completed":
        cube_start -= onp.timedelta64(round(shape[0] / fps * 1e9), "ns")
    indices = onp.arange(shape[0])
    times = cube_start + onp.rint(indices / fps * 1e9).astype("timedelta64[ns]")
    first = 0 if start is None else int(onp.searchsorted(times, utc64(start)))
    stop = shape[0] if end is None else int(onp.searchsorted(times, utc64(end)))
    return CubeSelection(path, hdu_index, shape, header, filename_time,
                         cube_start, fps, first, stop)


def index_fits(files, *, start=None, end=None, timestamp="start"):
    """Index explicit paths, retaining only cubes with selected frame times."""
    if start is not None and end is not None and utc64(end) <= utc64(start):
        raise ValueError("end must be after start")
    cubes = [inspect_cube(p, timestamp=timestamp, start=start, end=end)
             for p in dict.fromkeys(map(Path, files))]
    return sorted((c for c in cubes if c.first < c.stop),
                  key=lambda c: (c.cube_start, str(c.path)))


def read_cube_batches(cube, batch_size=256):
    """Yield (original frame indices, pixels) using lazy FITS section reads.

    Section access also avoids decompressing an entire CompImageHDU cube.
    """
    if not isinstance(batch_size, (int, onp.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    with fits.open(cube.path, memmap=False) as hdus:
        image = hdus[cube.hdu_index]
        for first in range(cube.first, cube.stop, batch_size):
            stop = min(first + batch_size, cube.stop)
            yield onp.arange(first, stop), onp.asarray(image.section[first:stop])
