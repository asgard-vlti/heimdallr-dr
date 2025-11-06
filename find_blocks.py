"""
Go through a directory and find all blocks of files that have filename timestamps
within a certain time delta of each other.

files are in the locked/ directory and have names like:
hei_k2_T04:03:21.651.fits 

There are 2 blocks:

03:55:46.627 04:05:56.693
04:12:11.740 04:20:21.790
"""
from glob import glob
# Minimum delta time to define a block
delta_time = 20.0  

all_files = sorted(glob('locked/hei_k2_T*.fits'))
from datetime import datetime
import os
import re

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

# Extract times (in seconds since midnight) and their string forms
times = []
for f in all_files:
    parsed = _parse_time_seconds_and_str(f)
    if parsed:
        times.append(parsed)

if not times:
    raise SystemExit(0)

# Ensure chronological order just in case sorting by filename wasn't sufficient
times.sort(key=lambda x: x[0])

# Build blocks where gaps between consecutive files exceed delta_time
blocks = []
start_idx = 0
last_sec = times[0][0]
for i in range(1, len(times)):
    sec = times[i][0]
    if sec - last_sec > delta_time:
        blocks.append((start_idx, i - 1))
        start_idx = i
    last_sec = sec
blocks.append((start_idx, len(times) - 1))

# Print first and last timestamp for each block
for s, e in blocks:
    print(f"{times[s][1]} {times[e][1]}")