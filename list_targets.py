"""
List the targets observed in a night from the VLTI TCS header dumps.

Reads the TCSHDR_*.iss files (FITS headers) in a headers directory, groups
consecutive headers that share a pointing, identifies each pointing with
Simbad (preferring HD numbers, then HIP, then the Simbad main identifier),
and prints target name, RA, Dec, UT and Paranal local time.

The UT date is taken from the YYYYMMDD directory in the path, e.g.
/Users/mireland/data/mimir/20260809/headers -> UT 2026-08-09 (this matches the
UTDATE keyword in the data files and the MJD in the TCS headers; data in a
directory start at 00:00 UT). Header UTC values are seconds-of-day on that UT
date. Local time uses America/Santiago, so Chilean summer time is handled
automatically from the full UT date+time (local evening date is the day before).

Usage:
    python list_targets.py /Users/mireland/data/mimir/20260809/headers
    python list_targets.py <dir> --per-file     # one row per header file
    python list_targets.py <dir> --csv out.csv  # also write a CSV
    python list_targets.py <dir> --obs          # print an `obs` list for process_block.py

Importable:
    from list_targets import get_pointings, target_at
"""
import argparse
import csv
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from glob import glob
from zoneinfo import ZoneInfo

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

LOCAL_TZ = ZoneInfo("America/Santiago")  # Paranal civil time
SEARCH_RADIUS_ARCSEC = 30.0
# Headers within this many arcsec are treated as the same pointing.
SAME_POINTING_ARCSEC = 5.0

_simbad_cache = {}
_good_server = None


# ---------------------------------------------------------------------------
# Date / time helpers
# ---------------------------------------------------------------------------
def ut_date_from_path(path):
    """Find a YYYYMMDD component in the path (searching from the end): the UT date."""
    for part in reversed(os.path.abspath(path).split(os.sep)):
        m = re.fullmatch(r"(20\d{2})(\d{2})(\d{2})", part)
        if m:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    raise ValueError(f"No YYYYMMDD date found in path: {path}")


def ut_datetime(ut_date, utc_seconds):
    """Full UT datetime from the UT date and UTC seconds-of-day."""
    return datetime(ut_date.year, ut_date.month, ut_date.day, tzinfo=timezone.utc) + timedelta(
        seconds=float(utc_seconds)
    )


def _fmt_time(dt):
    return dt.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Simbad
# ---------------------------------------------------------------------------
def _simbad_tap(adql):
    """Run an ADQL query, falling back through the Simbad mirrors."""
    from astroquery.simbad import conf
    from astroquery.simbad.core import SimbadClass

    global _good_server
    original = conf.server
    servers = [original] + [s for s in conf.servers_list if s != original]
    if _good_server in servers:  # try the last server that worked first
        servers.remove(_good_server)
        servers.insert(0, _good_server)
    errors = []
    try:
        for server in servers:
            conf.server = server
            try:
                result = SimbadClass().query_tap(adql)
                _good_server = server
                return result
            except Exception as e:  # network / service errors
                errors.append(f"{server}: {e.__class__.__name__}")
    finally:
        conf.server = original
    raise RuntimeError("All Simbad servers failed: " + "; ".join(errors))


def _preferred_name(main_id, ids):
    idlist = [s.strip() for s in str(ids).split("|")]
    for prefix in ("HD ", "HIP "):
        for name in idlist:
            if name.startswith(prefix):
                return " ".join(name.split()).replace(" ", "")
    return " ".join(str(main_id).split())


def simbad_name(ra, dec, radius_arcsec=SEARCH_RADIUS_ARCSEC):
    """Return (name, separation_arcsec, kmag) for the brightest object near ra, dec."""
    key = (round(ra, 5), round(dec, 5))
    if key in _simbad_cache:
        return _simbad_cache[key]
    adql = f"""
        SELECT b.main_id, i.ids, k.flux AS kmag, v.flux AS vmag,
               DISTANCE(POINT('ICRS', b.ra, b.dec), POINT('ICRS', {ra}, {dec})) * 3600 AS sep
        FROM basic AS b
        JOIN ids AS i ON i.oidref = b.oid
        LEFT JOIN flux AS k ON k.oidref = b.oid AND k.filter = 'K'
        LEFT JOIN flux AS v ON v.oidref = b.oid AND v.filter = 'V'
        WHERE CONTAINS(POINT('ICRS', b.ra, b.dec),
                       CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec / 3600.0})) = 1
    """
    try:
        t = _simbad_tap(adql)
    except RuntimeError as e:
        print(f"WARNING: Simbad lookup failed for {ra:.5f} {dec:.5f}: {e}", file=sys.stderr)
        t = []
    if len(t) == 0:
        result = ("UNKNOWN", float("nan"), float("nan"))
    else:
        def brightness(row):
            # Brightest in K; fall back to V; objects with no photometry last.
            for col in ("kmag", "vmag"):
                val = row[col]
                if val is not None and not hasattr(val, "mask") and val == val:
                    return float(val)
            return 99.0

        best = min(t, key=lambda r: (brightness(r), float(r["sep"])))
        kmag = best["kmag"]
        kmag = float(kmag) if kmag is not None and not hasattr(kmag, "mask") else float("nan")
        result = (_preferred_name(best["main_id"], best["ids"]), float(best["sep"]), kmag)
    _simbad_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Header reading
# ---------------------------------------------------------------------------
def read_headers(header_dir):
    """Return a time-sorted list of dicts, one per .iss header file."""
    ut_date = ut_date_from_path(header_dir)
    rows = []
    for f in glob(os.path.join(header_dir, "*.iss")):
        h = fits.Header.fromfile(f)
        if "RA" not in h or "DEC" not in h or "UTC" not in h:
            continue
        ut = ut_datetime(ut_date, h["UTC"])
        # Sanity check against an MJD in the header, if there is one.
        mjd = h.get("ESO ISS AT1 ALT TIME START")
        if mjd is not None:
            mjd_dt = datetime(1858, 11, 17, tzinfo=timezone.utc) + timedelta(days=float(mjd))
            if abs((mjd_dt - ut).total_seconds()) > 3600:
                print(f"WARNING: {os.path.basename(f)} UT {ut} disagrees with header MJD "
                      f"{mjd_dt}; check the directory date.", file=sys.stderr)
        rows.append(dict(file=os.path.basename(f), ra=float(h["RA"]), dec=float(h["DEC"]), ut=ut))
    rows.sort(key=lambda r: r["ut"])
    return rows


def get_pointings(header_dir, per_file=False):
    """Group consecutive headers with the same pointing and identify them.

    Returns a list of dicts with keys: target, ra, dec, coord, sep, kmag,
    ut_start, ut_end, local_start, local_end, files.
    """
    rows = read_headers(header_dir)
    groups = []
    for r in rows:
        c = SkyCoord(r["ra"], r["dec"], unit="deg")
        if (not per_file and groups
                and groups[-1]["coord"].separation(c) < SAME_POINTING_ARCSEC * u.arcsec):
            groups[-1]["ut_end"] = r["ut"]
            groups[-1]["files"].append(r["file"])
        else:
            groups.append(dict(ra=r["ra"], dec=r["dec"], coord=c, ut_start=r["ut"],
                               ut_end=r["ut"], files=[r["file"]]))
    for g in groups:
        g["target"], g["sep"], g["kmag"] = simbad_name(g["ra"], g["dec"])
        g["local_start"] = g["ut_start"].astimezone(LOCAL_TZ)
        g["local_end"] = g["ut_end"].astimezone(LOCAL_TZ)
    return groups


def target_at(pointings, ut):
    """Return the pointing that was active at datetime `ut` (the most recent
    one starting at or before it), or None if before the first pointing."""
    current = None
    for p in pointings:
        if p["ut_start"] <= ut:
            current = p
        else:
            break
    return current


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("header_dir", help="Directory with TCSHDR_*.iss files (path must contain the UT date as YYYYMMDD)")
    parser.add_argument("--per-file", action="store_true", help="One row per header file instead of per pointing")
    parser.add_argument("--csv", help="Also write the table to this CSV file")
    parser.add_argument("--obs", action="store_true",
                        help="Print a Python `obs` list of [target, UT start, UT end] for process_block.py")
    args = parser.parse_args()

    pointings = get_pointings(args.header_dir, per_file=args.per_file)
    if not pointings:
        sys.exit(f"No .iss headers found in {args.header_dir}")

    ut_date = ut_date_from_path(args.header_dir)
    print(f"UT date {ut_date}, local night of {pointings[0]['local_start'].date()} "
          f"(America/Santiago, UTC{pointings[0]['local_start'].strftime('%z')[:3]})")
    header = (f"{'Target':<16} {'RA (J2000)':<12} {'Dec (J2000)':<12} {'UT date':<10} "
              f"{'UT start':<8} {'UT end':<8} {'Local start':<11} {'Local end':<9} {'K':>5} {'sep':>5}  Files")
    print(header)
    print("-" * len(header))
    for p in pointings:
        ra = p["coord"].ra.to_string(unit=u.hourangle, sep=":", precision=1, pad=True)
        dec = p["coord"].dec.to_string(unit=u.deg, sep=":", precision=0, alwayssign=True, pad=True)
        print(f"{p['target']:<16} {ra:<12} {dec:<12} {p['ut_start'].date()!s:<10} "
              f"{_fmt_time(p['ut_start']):<8} {_fmt_time(p['ut_end']):<8} "
              f"{_fmt_time(p['local_start']):<11} {_fmt_time(p['local_end']):<9} "
              f"{p['kmag']:5.2f} {p['sep']:5.1f}  {','.join(p['files'])}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["target", "ra_deg", "dec_deg", "ut_start", "ut_end",
                        "local_start", "local_end", "kmag", "sep_arcsec", "files"])
            for p in pointings:
                w.writerow([p["target"], f"{p['ra']:.6f}", f"{p['dec']:.6f}",
                            p["ut_start"].isoformat(), p["ut_end"].isoformat(),
                            p["local_start"].isoformat(), p["local_end"].isoformat(),
                            f"{p['kmag']:.3f}", f"{p['sep']:.2f}", " ".join(p["files"])])
        print(f"Wrote {args.csv}")

    if args.obs:
        print("\nobs = [")
        for p in pointings:
            print(f'    ["{p["target"]}", "{_fmt_time(p["ut_start"])}", "{_fmt_time(p["ut_end"])}"],')
        print("]")


if __name__ == "__main__":
    main()
