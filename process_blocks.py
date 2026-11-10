"""
Semi-automatic processing of a night of Heimdallr data.

1. find_blocks.find_blocks() splits the data files into contiguous blocks.
2. list_targets.get_pointings() reads the TCS headers and identifies each
   pointing with Simbad.
3. Each block is assigned the target that was being pointed at when it started,
   and suspicious blocks are flagged:
     SHORT     fewer than --min-files files
     SPANS     a new pointing starts part way through the block (use --split)
     STALE     block starts more than --stale minutes after the last header of
               its pointing (maybe an un-headed target or a calibration)
     NOTARGET  block starts before the first pointing
4. process_block.process_block() is run on the chosen blocks and V^2 and closure
   phases are appended to a CSV as each block finishes.

Usage:
    python process_blocks.py /Users/mireland/data/mimir/20260809 --list
    python process_blocks.py <night_dir>                    # ask before each block
    python process_blocks.py <night_dir> --yes              # all unflagged blocks, no questions
    python process_blocks.py <night_dir> --blocks 3,5-8     # only these block numbers
    python process_blocks.py <night_dir> --target HD222805  # only this target (repeatable)
    python process_blocks.py <night_dir> --split            # split blocks at pointing changes

The night directory is named by UT date (YYYYMMDD) and contains headers/ plus
the data files, either directly or in compressed/ or locked/.
"""
import argparse
import csv
import math
import os
import sys
from datetime import datetime, timedelta, timezone

from find_blocks import DELTA_TIME, find_blocks
from list_targets import get_pointings, target_at, ut_date_from_path

# Must match process_block.bl2beam and its cp2bl closure triangles.
BASELINES = ["12", "13", "14", "23", "24", "34"]
TRIANGLES = ["123", "124", "134", "234"]  # telescopes in each cp2bl triangle
SPAN_GRACE_S = 60  # a header this soon after block start belongs to the block


def find_data_dir(night_dir):
    for sub in ("compressed", "locked", ""):
        d = os.path.join(night_dir, sub)
        if os.path.isdir(d) and any(f.startswith("hei_k1_T") for f in os.listdir(d)):
            return d
    sys.exit(f"No hei_k1_T*.fits files found in {night_dir}, {night_dir}/compressed or {night_dir}/locked")


def _hms(sec):
    sec = int(sec) % 86400
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def _ut(ut_date, sec):
    return datetime(ut_date.year, ut_date.month, ut_date.day, tzinfo=timezone.utc) + timedelta(seconds=sec)


def _pointing_starts_inside(pointings, ut_date, times):
    """Pointings that start strictly within this run of file times."""
    t0 = _ut(ut_date, times[0][0] + SPAN_GRACE_S)
    t1 = _ut(ut_date, times[-1][0])
    return [p for p in pointings if t0 < p["ut_start"] <= t1]


def _split_times(times, ut_date, pointings):
    """Split a block's file times at every pointing start that falls inside it."""
    pieces = [times]
    for p in _pointing_starts_inside(pointings, ut_date, times):
        last = pieces.pop()
        before = [t for t in last if _ut(ut_date, t[0]) < p["ut_start"]]
        after = [t for t in last if _ut(ut_date, t[0]) >= p["ut_start"]]
        pieces += [x for x in (before, after) if x]
    return pieces


def build_block_table(night_dir, data_dir, delta, min_files, stale_min, split):
    ut_date = ut_date_from_path(night_dir)
    pointings = get_pointings(os.path.join(night_dir, "headers"))
    raw = find_blocks(data_dir, delta)

    runs = []
    for b in raw:
        runs += _split_times(b["times"], ut_date, pointings) if split else [b["times"]]

    table = []
    for times in runs:
        start = _ut(ut_date, times[0][0])
        p = target_at(pointings, start + timedelta(seconds=SPAN_GRACE_S))
        flags = []
        if len(times) < min_files:
            flags.append("SHORT")
        if _pointing_starts_inside(pointings, ut_date, times):
            flags.append("SPANS")
        if p is None:
            flags.append("NOTARGET")
        elif (start - p["ut_end"]).total_seconds() > stale_min * 60:
            flags.append("STALE")
        table.append(dict(
            idx=len(table),
            target=p["target"] if p else "UNKNOWN",
            tstart=_hms(math.floor(times[0][0])),       # rounded outwards so
            tstop=_hms(math.ceil(times[-1][0])),        # process_block keeps the edge files
            nfiles=len(times),
            local=start.astimezone(pointings[0]["local_start"].tzinfo) if pointings else start,
            flags=flags,
        ))
    return table


def print_table(table):
    print(f"{'#':>3} {'Target':<16} {'UT start':<8} {'UT stop':<8} {'Local':<8} {'Files':>5}  Flags")
    for b in table:
        print(f"{b['idx']:3d} {b['target']:<16} {b['tstart']:<8} {b['tstop']:<8} "
              f"{b['local'].strftime('%H:%M:%S'):<8} {b['nfiles']:5d}  {' '.join(b['flags'])}")


def parse_index_list(spec):
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part.strip():
            out.add(int(part))
    return out


def csv_header():
    return (["target", "ut_date", "tstart", "tstop", "nfiles", "flags"]
            + [f"V2_K1_{b}" for b in BASELINES] + [f"V2_K2_{b}" for b in BASELINES]
            + [f"CP_K1_{t}" for t in TRIANGLES] + [f"CP_K2_{t}" for t in TRIANGLES])


def main():
    parser = argparse.ArgumentParser(description="Semi-automatic processing of a night of Heimdallr data.")
    parser.add_argument("night_dir", help="Night directory named YYYYMMDD (UT), containing headers/")
    parser.add_argument("--data-dir", help="Data file directory (default: auto-detect compressed/, locked/ or night_dir)")
    parser.add_argument("--list", action="store_true", help="Only print the block table")
    parser.add_argument("--yes", "-y", action="store_true", help="Process all selected unflagged blocks without asking")
    parser.add_argument("--blocks", help="Block numbers to process, e.g. 3,5-8 (flagged blocks allowed)")
    parser.add_argument("--target", action="append", help="Only process this target (repeatable)")
    parser.add_argument("--split", action="store_true", help="Split blocks at pointing changes")
    parser.add_argument("--delta", type=float, default=DELTA_TIME, help="Gap (s) that starts a new block")
    parser.add_argument("--min-files", type=int, default=6, help="Blocks with fewer files are flagged SHORT")
    parser.add_argument("--stale", type=float, default=15.0,
                        help="Flag blocks starting this many minutes after the last header (STALE)")
    parser.add_argument("--out", help="Output CSV (default: heimdallr_<YYYYMMDD>.csv); results are appended")
    args = parser.parse_args()

    night_dir = os.path.abspath(args.night_dir)
    data_dir = args.data_dir or find_data_dir(night_dir)
    ut_date = ut_date_from_path(night_dir)
    print(f"Data: {data_dir}\nLooking up targets in Simbad...")
    table = build_block_table(night_dir, data_dir, args.delta, args.min_files, args.stale, args.split)
    print_table(table)
    if args.list:
        return

    # Select blocks
    chosen = table
    if args.blocks:
        wanted = parse_index_list(args.blocks)
        chosen = [b for b in chosen if b["idx"] in wanted]
    if args.target:
        names = {t.replace(" ", "").upper() for t in args.target}
        chosen = [b for b in chosen if b["target"].replace(" ", "").upper() in names]
    if not args.blocks:  # explicitly listed blocks may be flagged; otherwise ask/skip
        if args.yes:
            chosen = [b for b in chosen if not b["flags"]]
    if not chosen:
        sys.exit("No blocks selected.")

    # Import late: process_block pulls in matplotlib and sets interactive mode.
    import matplotlib.pyplot as plt
    from process_block import process_block

    out_path = args.out or f"heimdallr_{ut_date.strftime('%Y%m%d')}.csv"
    new_file = not os.path.exists(out_path)
    fh = open(out_path, "a", newline="")
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(csv_header())

    ask = not args.yes
    n_ok, failed = 0, []
    try:
        for b in chosen:
            label = f"#{b['idx']} {b['target']} {b['tstart']}-{b['tstop']} ({b['nfiles']} files)"
            if b["flags"]:
                label += f" [{' '.join(b['flags'])}]"
            if ask:
                default = "n" if b["flags"] else "y"
                reply = input(f"Process {label}? [y/n/a(ll remaining)/q] ({default}): ").strip().lower() or default
                if reply.startswith("q"):
                    break
                if reply.startswith("a"):
                    ask = False
                elif not reply.startswith("y"):
                    continue
            print(f"\n=== Processing {label}")
            try:
                res = process_block(b["tstart"], b["tstop"], os.path.join(data_dir, ""), target=b["target"])
            except Exception as e:
                print(f"FAILED {label}: {e}")
                failed.append(label)
                continue
            writer.writerow([b["target"], ut_date.isoformat(), b["tstart"], b["tstop"], b["nfiles"], " ".join(b["flags"])]
                            + [f"{v:.4f}" for v in res["V2_k1"]] + [f"{v:.4f}" for v in res["V2_k2"]]
                            + [f"{v:.2f}" for v in res["cp_k1"]] + [f"{v:.2f}" for v in res["cp_k2"]])
            fh.flush()
            n_ok += 1
            plt.pause(0.1)  # let the diagnostic plots draw before the next prompt
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        fh.close()

    print(f"\n{n_ok} block(s) processed, results appended to {out_path}")
    if failed:
        print(f"{len(failed)} failed:")
        for f in failed:
            print("  " + f)


if __name__ == "__main__":
    main()
