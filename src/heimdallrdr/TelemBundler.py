import datetime
import pathlib
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from ._timestamps import utc_timestamp


def load_ttmet(fname, start_time=None, end_time=None):
    """
    Loads a ttmet file into a pd dataframe

    time tx ty mx my
    1775184464.9896 0.126 -0.157 0.064 -0.025
    prev 0.202 -0.146 0.066 -0.027
    prev 0.307 -0.120 0.068 -0.027
    prev 0.304 -0.272 0.071 -0.030
    1775184464.9944 0.478 -0.337 0.075 -0.033
    prev 0.518 -0.337 0.079 -0.036
    prev 0.548 -0.186 0.084 -0.037
    prev 0.468 0.256 0.088 -0.035
    prev 0.523 0.458 0.092 -0.030
    1775184464.9994 0.441 0.562 0.095 -0.024
    prev 0.219 0.657 0.097 -0.017
    prev 0.105 0.376 0.097 -0.013
    prev -0.033 0.132 0.095 -0.012
    prev -0.219 0.031 0.092 -0.011
    1775184465.0047 -0.217 0.092 0.089 -0.010

    ...
    in cases of "prev" in time, use the next timestamp and interpolate linearly

    Optional start_time/end_time bounds can be supplied to only materialize rows
    in the requested interval. When bounds are provided, the parser streams the
    file and avoids storing rows outside the window. Naive datetime bounds are
    interpreted as UTC; numeric bounds are Unix seconds.
    """

    # Accept either a path-like or a string
    p = pathlib.Path(fname)

    # Convert bounds to Unix seconds independently of the machine timezone.
    start_time_sec = (
        utc_timestamp(start_time) if start_time is not None else None
    )
    end_time_sec = (
        utc_timestamp(end_time) if end_time is not None else None
    )

    data = {"time": [], "tx": [], "ty": [], "mx": [], "my": []}

    # Count lines first so tqdm can display a real progress bar with completion.
    with p.open("r") as fh:
        total_lines = sum(1 for _ in fh)

    with p.open("r") as fh:
        prev_time = None
        # to hold rows with "prev" time until we get a real timestamp to assign them
        # (then interpolate linearly if there are multiple "prev" rows in a row)
        prev_stack = []

        for ln in tqdm(fh, total=total_lines, desc="load_ttmet", unit="line"):
            s = ln.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 5:
                continue

            time_str, tx_str, ty_str, mx_str, my_str = parts[:5]

            if time_str == "prev":
                if prev_time is None:
                    continue  # skip "prev" rows until we get a real timestamp
                else:
                    prev_stack.append((tx_str, ty_str, mx_str, my_str))
            else:
                try:
                    t = float(time_str)
                except Exception:
                    continue

                # Now add the current real timestamp row
                if (start_time_sec is not None and t < start_time_sec) or (
                    end_time_sec is not None and t > end_time_sec
                ):
                    continue  # skip rows outside the requested time window

                # If we have "prev" rows stacked, assign them timestamps by interpolating between prev_time and t
                if prev_stack:
                    n_prev = len(prev_stack)
                    for i, (tx_s, ty_s, mx_s, my_s) in enumerate(prev_stack):
                        interp_time = prev_time + (t - prev_time) * (i + 1) / (
                            n_prev + 1
                        )
                        data["time"].append(
                            datetime.datetime.fromtimestamp(
                                interp_time, datetime.timezone.utc
                            )
                        )
                        data["tx"].append(float(tx_s))
                        data["ty"].append(float(ty_s))
                        data["mx"].append(float(mx_s))
                        data["my"].append(float(my_s))
                    prev_stack = []

                data["time"].append(
                    datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
                )
                try:
                    data["tx"].append(float(tx_str))
                    data["ty"].append(float(ty_str))
                    data["mx"].append(float(mx_str))
                    data["my"].append(float(my_str))
                except Exception:
                    data["tx"].append(np.nan)
                    data["ty"].append(np.nan)
                    data["mx"].append(np.nan)
                    data["my"].append(np.nan)

                prev_time = t

    df = pd.DataFrame(data)
    return df
