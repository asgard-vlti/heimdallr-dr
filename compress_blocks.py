import argparse
import os
import shutil
import tarfile
from pathlib import Path
from _script_timestamps import parse_time_seconds_and_str


def parse_time(filename):
    """
    Extracts time string HHMMSS from supported camera filename formats.
    """
    parsed = parse_time_seconds_and_str(filename)
    if parsed is not None:
        return parsed[1].split(".", 1)[0].replace(":", "")
    return None


def get_subset_prefix(subset):
    """
    Maps user subset argument to filename prefix.
    """
    if subset.startswith("baldr"):
        return subset
    elif subset.startswith("hei_k"):
        return subset
    elif subset == "hei1":
        return "hei_k1"
    elif subset == "hei2":
        return "hei_k2"
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Grab, move, and compress FITS files from /data/{date}/."
    )
    parser.add_argument("date", help="Date string for the folder, e.g., 20231024")
    parser.add_argument(
        "--data_subsets",
        nargs="+",
        default=["all"],
        help="List of subsets: 'baldr1', 'baldr', 'hei', 'all', etc.",
    )
    parser.add_argument("--time_start", default=None, help="Start time (HHMMSS)")
    parser.add_argument("--time_end", default=None, help="End time (HHMMSS)")
    parser.add_argument(
        "--destination", default=".", help="Destination folder for compressed file"
    )

    args = parser.parse_args()

    source_dir = Path(f"/data/{args.date}")
    if not source_dir.exists():
        print(f"Error: Directory {source_dir} does not exist.")
        return

    # Expand subsets
    subsets_to_process = []
    if "all" in args.data_subsets:
        subsets_to_process = [
            "baldr1",
            "baldr2",
            "baldr3",
            "baldr4",
            "hei_k1",
            "hei_k2",
        ]
    else:
        for sub in args.data_subsets:
            if sub == "baldr":
                subsets_to_process.extend(["baldr1", "baldr2", "baldr3", "baldr4"])
            elif sub == "hei":
                subsets_to_process.extend(["hei_k1", "hei_k2"])
            else:
                prefix = get_subset_prefix(sub)
                if prefix:
                    subsets_to_process.append(prefix)

    # Gather files
    files_to_compress = []
    for f in source_dir.iterdir():
        if not f.is_file() or not f.name.endswith(".fits"):
            continue

        # Check prefix
        valid_prefix = False
        for prefix in subsets_to_process:
            if f.name.startswith(prefix + "_"):
                valid_prefix = True
                break

        if not valid_prefix:
            continue

        # Check time
        f_time = parse_time(f.name)
        if f_time is None:
            continue

        if args.time_start and f_time < args.time_start.replace("T", ""):
            continue
        if args.time_end and f_time > args.time_end.replace("T", ""):
            continue

        files_to_compress.append(f)

    if not files_to_compress:
        print("No files found matching the criteria.")
        return

    dest_dir = Path(args.destination)
    dest_dir.mkdir(parents=True, exist_ok=True)

    archive_name = dest_dir / f"data_{args.date}.tar.gz"

    print(f"Compressing {len(files_to_compress)} files into {archive_name}...")

    with tarfile.open(archive_name, "w:gz") as tar:
        for f in files_to_compress:
            tar.add(f, arcname=f.name)

    print("Done!")


if __name__ == "__main__":
    main()
