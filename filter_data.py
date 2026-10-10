"""
This will read in telemetry data from a file, removing rows where all of the
final values are zero. Because the file is large, we process it in chunks and
write the filtered output to a new file.

Each line is space-separated and the header is a comment beginning with '#'.
"""

from __future__ import annotations

import os

import pandas as pd


def filter_ft_performance_file(infile, outfile=None, dm_only=True, chunksize=10**6):
    """Filter ft_performance log rows while preserving the timestamp column."""
    if outfile is None:
        suffix = "_dm.log" if dm_only else "_cleaned.log"
        outfile = infile.replace(".log", suffix)

    if os.path.exists(outfile):
        os.remove(outfile)

    header_line = None
    with open(infile, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                header_line = line.rstrip("\n")
                break

    if header_line is not None:
        with open(outfile, "w", encoding="utf-8") as handle:
            handle.write(header_line + "\n")

    for chunk in pd.read_csv(infile, sep=r"\s+", comment="#", header=None, chunksize=chunksize):
        if chunk.empty:
            continue

        chunk = chunk.dropna(how="all")
        if chunk.empty:
            continue

        chunk = chunk.dropna(axis=0, thresh=30)
        if chunk.empty:
            continue

        keep = (~(chunk.iloc[:, -4:] == 0).all(axis=1)) if dm_only else (~(chunk.iloc[:, -8:] == 0).all(axis=1))
        filtered_chunk = chunk.loc[keep]
        if filtered_chunk.empty:
            continue

        filtered_chunk.to_csv(outfile, mode="a", header=False, index=False, sep=" ")

    return outfile


def main():
    ddir = "/Users/mireland/data/mimir/"
    infile = ddir + "20260914/ft_performance_20260914T221159.log"
    #ddir = "/Users/mireland/data/heimdallr/telemetry/2509/"
    #infile = ddir + "20250916/ft_performance_20250917T021019.log"
    filter_ft_performance_file(infile, dm_only=True)


if __name__ == "__main__":
    main()
