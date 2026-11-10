"""
Go through a directory and find all blocks of files that have filename timestamps
within a certain time delta of each other.

Files have names like:
hei_k2_T04:03:21.651.fits

Example output (one line per block, first and last timestamp):

03:55:46.627 04:05:56.693
04:12:11.740 04:20:21.790

Usage:
    python find_blocks.py [data_dir] [--delta 20] [--pattern 'hei_k2_T*.fits']

data_dir defaults to locked/ (the original behaviour).

Importable:
    from find_blocks import find_blocks
"""
import argparse
import os
import re
from datetime import datetime
from glob import glob

# Minimum gap (s) between consecutive files that defines a new block
DELTA_TIME = 20.0
DEFAULT_PATTERN = "hei_k2_T*.fits"


def _parse_time_seconds_and_str(path: str):
    base = os.path.basename(path)
    m = re.search(r'T(\d{2}:\d{2}:\d{2}(?:\.\d+)?)', base)
    if not m:
        return None
    tstr = m.group(1)
    try:
        dt = datetime.strptime(tstr, '%H:%M:%S.%f')
    except ValueError:
        dt = datetime.strptime(tstr, '%H:%M:%S')
    sec = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    return sec, tstr


def find_blocks(data_dir="locked", delta_time=DELTA_TIME, pattern=DEFAULT_PATTERN):
    """Split the files in data_dir into blocks separated by gaps > delta_time.

    Returns a list of dicts with keys:
        tstart, tstop : 'HH:MM:SS.fff' strings of the first/last file
        start_sec, stop_sec : seconds since UT midnight
        nfiles : number of files in the block
        times : list of (seconds, 'HH:MM:SS.fff') for every file in the block
    """
    times = []
    for f in glob(os.path.join(data_dir, pattern)):
        parsed = _parse_time_seconds_and_str(f)
        if parsed:
            times.append(parsed)
    if not times:
        return []

    # Sort chronologically (filename order is the same, but be safe)
    times.sort(key=lambda x: x[0])

    # Indices (start, end) of blocks, split where the gap exceeds delta_time
    edges = []
    start_idx = 0
    for i in range(1, len(times)):
        if times[i][0] - times[i - 1][0] > delta_time:
            edges.append((start_idx, i - 1))
            start_idx = i
    edges.append((start_idx, len(times) - 1))

    return [dict(tstart=times[s][1], tstop=times[e][1],
                 start_sec=times[s][0], stop_sec=times[e][0], nfiles=e - s + 1,
                 times=times[s:e + 1])
            for s, e in edges]


def main():
    parser = argparse.ArgumentParser(description="Find blocks of contiguous data files.")
    parser.add_argument("data_dir", nargs="?", default="locked", help="Directory of data files (default: locked)")
    parser.add_argument("--delta", type=float, default=DELTA_TIME,
                        help=f"Gap in seconds that starts a new block (default {DELTA_TIME})")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help=f"File glob (default {DEFAULT_PATTERN})")
    parser.add_argument("-v", "--verbose", action="store_true", help="Also print number of files per block")
    args = parser.parse_args()

    for b in find_blocks(args.data_dir, args.delta, args.pattern):
        if args.verbose:
            print(f"{b['tstart']} {b['tstop']} {b['nfiles']:4d}")
        else:
            print(f"{b['tstart']} {b['tstop']}")


if __name__ == "__main__":
    main()
