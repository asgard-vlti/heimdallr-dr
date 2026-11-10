"""
Plan target groups for a night at the VLTI (Paranal) with Heimdallr.

Selects targets that
  * come from a first-priority list (default targets_P117g.csv) or a
    second-priority list (default targets_P117_AS.csv),
  * have not been observed before (not within --prev-match arcsec of any entry
    in prev_tgts.csv, e.g. a CSV written by list_targets.py --csv),
  * are above --min-alt (default 35 deg) for at least --min-hours during the
    astronomical night (Sun below --sun-alt, default -18 deg) of --date
    (default: tomorrow, as the local evening date).

These targets are then grouped into groups of up to 3 targets in which every pair
is within --max-sep (default 10 deg) and all members are simultaneously
observable for at least --min-hours. The grouping is optimal in the sense of an
exact integer program: it minimises the number of groups (so triples are
preferred) and, among equal solutions, the total separation within groups.

Only elevation is modelled: delay-line and shadowing limits of the large AT
configuration are NOT included.

Outputs (one row per group) a CSV, sorted by priority (groups containing a
first-priority target first) and then by start of the common window, plus an
RA-Dec plot with colour-coded groups and previously observed targets in grey.

Usage:
    python plan_targets.py
    python plan_targets.py --date 2026-11-11 --out groups_20261111
"""
import argparse
import itertools
import re
from datetime import date, datetime, timedelta

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from zoneinfo import ZoneInfo

# VLTI platform, Cerro Paranal (hard-coded so no network lookup is needed).
PARANAL = EarthLocation.from_geodetic(lon=-70.4045 * u.deg, lat=-24.6268 * u.deg, height=2635 * u.m)
LOCAL_TZ = ZoneInfo("America/Santiago")
LARGE_CONFIG = "A0-G1-J2-K0"  # stations used for "large" on 2026-08-09 (from the TCS headers)
TIME_STEP_MIN = 5


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
def clean_name(name):
    """'HD   2221' -> 'HD 2221', '*  45 Psc' -> '45 Psc'."""
    name = re.sub(r"^\*+\s*", "", name.strip())
    return " ".join(name.split())


# Catalogues whose "<prefix> <number>" is unambiguous with the space removed.
_JOINABLE = ("HD", "HIP", "HR", "TYC", "GJ")


def obs_name(name):
    """Single-token name for the OB-creation script.

    'HD 225214' -> 'HD225214', 'TYC 594-550-1' -> 'TYC594-550-1';
    anything else keeps the space as an underscore, since joining would be
    ambiguous: 'CD-41 339' -> 'CD-41_339', '45 Psc' -> '45_Psc',
    'nu. Oph' -> 'nu_Oph' (Simbad's trailing '.' on an unnumbered Greek letter dropped).
    """
    parts = name.split()
    if len(parts) == 2 and parts[0] in _JOINABLE:
        return parts[0] + parts[1]
    parts = [re.sub(r"^([a-z]+)\.$", r"\1", p) for p in parts]
    return "_".join(parts)


def read_target_list(path, priority):
    """Read a 'Name, RA, Dec, Mag' list (RA/Dec sexagesimal)."""
    rows = []
    with open(path) as fh:
        next(fh)  # header
        for line in fh:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3 or not parts[0]:
                continue
            rows.append(dict(name=clean_name(parts[0]), ra=parts[1], dec=parts[2],
                             mag=parts[3] if len(parts) > 3 else "", priority=priority))
    coords = SkyCoord([r["ra"] for r in rows], [r["dec"] for r in rows], unit=(u.hourangle, u.deg))
    for r, c in zip(rows, coords):
        r["coord"] = c
    return rows


def read_prev(path):
    """Read previously observed targets (list_targets.py CSV: target, ra_deg, dec_deg, ...)."""
    import csv
    seen = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            seen.setdefault(row["target"], (float(row["ra_deg"]), float(row["dec_deg"])))
    names = list(seen)
    if not names:
        return []
    coords = SkyCoord([seen[n][0] for n in names], [seen[n][1] for n in names], unit="deg")
    return [dict(name=n, coord=c) for n, c in zip(names, coords)]


def _match_mask(targets, others, radius_arcsec):
    """Boolean array: is each target within radius of any of `others`?"""
    if not targets or not others:
        return np.zeros(len(targets), bool)
    tc = SkyCoord([t["coord"] for t in targets])
    oc = SkyCoord([o["coord"] for o in others])
    _, sep, _ = tc.match_to_catalog_sky(oc)
    return sep < radius_arcsec * u.arcsec


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------
def night_grid(evening_date, sun_alt):
    """Times (astropy Time) at TIME_STEP_MIN spacing while the Sun is below sun_alt."""
    t0 = datetime(evening_date.year, evening_date.month, evening_date.day, 17, 0, tzinfo=LOCAL_TZ)
    times = Time(t0) + np.arange(0, 16 * 60, TIME_STEP_MIN) * u.min
    frame = AltAz(obstime=times, location=PARANAL)
    dark = get_sun(times).transform_to(frame).alt.deg < sun_alt
    if not dark.any():
        raise SystemExit("No dark time found for this date.")
    return times[dark]


def altitudes(targets, times):
    """Array (n_targets, n_times) of altitudes in degrees."""
    coords = SkyCoord([t["coord"] for t in targets])
    frame = AltAz(obstime=times, location=PARANAL)
    return coords[:, None].transform_to(frame[None, :]).alt.deg


def longest_run(mask):
    """(start_index, length) of the longest run of True in a 1D boolean array."""
    best, start, cur_start, cur = (0, 0), 0, None, 0
    for i, m in enumerate(mask):
        if m:
            if cur_start is None:
                cur_start, cur = i, 0
            cur += 1
            if cur > best[1]:
                best = (cur_start, cur)
        else:
            cur_start = None
    return best


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------
def candidate_groups(targets, up, max_sep, min_steps, max_size=3):
    """All groups of 1..max_size targets, pairwise within max_sep, with a common
    observable window of at least min_steps time steps."""
    coords = SkyCoord([t["coord"] for t in targets])
    n = len(targets)
    sep = coords[:, None].separation(coords[None, :]).deg
    close = sep <= max_sep
    groups = [((i,), 0.0) for i in range(n)]
    for size in range(2, max_size + 1):
        for combo in itertools.combinations(range(n), size):
            if not all(close[a, b] for a, b in itertools.combinations(combo, 2)):
                continue
            if longest_run(np.all(up[list(combo)], axis=0))[1] < min_steps:
                continue
            groups.append((combo, sum(sep[a, b] for a, b in itertools.combinations(combo, 2))))
    return groups


def optimal_partition(n, groups, max_sep):
    """Exact set partition: every target in exactly one group, minimising the
    number of groups, then the total pairwise separation."""
    # Separation term is always < 1 group's cost, so it only breaks ties.
    max_total_sep = 3 * max_sep * n + 1
    cost = np.array([1.0 + s / max_total_sep for _, s in groups])
    A = lil_matrix((n, len(groups)))
    for j, (members, _) in enumerate(groups):
        for i in members:
            A[i, j] = 1
    res = milp(cost, constraints=LinearConstraint(A.tocsr(), 1, 1),
               integrality=np.ones(len(groups)), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"Grouping optimisation failed: {res.message}")
    return [groups[j][0] for j in np.flatnonzero(res.x > 0.5)]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_groups(rows, targets, prev, ra_centre_h, title, path):
    def x_of(coord):
        # RA in hours, wrapped so the plot is contiguous around ra_centre_h
        return (coord.ra.hour - ra_centre_h + 12) % 24 - 12 + ra_centre_h

    fig, ax = plt.subplots(figsize=(16, 10))
    # tab20 without its two greys, so grey always means "previously observed"
    palette = [c for i, c in enumerate(plt.get_cmap("tab20").colors) if i not in (14, 15)]
    texts = []
    for p in prev:
        x, y = x_of(p["coord"]), p["coord"].dec.deg
        ax.plot(x, y, "o", color="0.6", ms=5, zorder=1)
        texts.append(ax.text(x, y, " " + p["name"], fontsize=6, color="0.5"))
    for k, row in enumerate(rows):
        col = palette[k % len(palette)]
        members = [targets[i] for i in row["members"]]
        xs = [x_of(t["coord"]) for t in members]
        ys = [t["coord"].dec.deg for t in members]
        if len(members) > 1:
            ax.plot(xs + [xs[0]] if len(members) == 3 else xs,
                    ys + [ys[0]] if len(members) == 3 else ys, "-", color=col, lw=1, zorder=2)
        for t, x, y in zip(members, xs, ys):
            ax.plot(x, y, "*" if t["priority"] == 1 else "o", color=col,
                    ms=11 if t["priority"] == 1 else 6, mec="k", mew=0.4, zorder=3)
            texts.append(ax.text(x, y, f" {t['name']} [G{row['group']}]", fontsize=6,
                                 color=tuple(0.75 * np.array(col))))  # darkened for legibility
    ticks = np.arange(np.floor(ax.get_xlim()[0]), np.ceil(ax.get_xlim()[1]) + 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t % 24)}h" for t in ticks])
    ax.invert_xaxis()  # RA increases to the left
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec (deg)")
    ax.grid(alpha=0.3)
    ax.plot([], [], "*", color="k", ms=10, label="priority 1 (P117g)")
    ax.plot([], [], "o", color="k", ms=6, label="priority 2 (P117_AS)")
    ax.plot([], [], "o", color="0.6", ms=5, label="previously observed")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(title)
    try:  # optional: pip install adjustText for non-overlapping labels
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="0.7", lw=0.5))
    except ImportError:
        pass
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--primary", default="targets_P117g.csv", help="First-priority target list")
    parser.add_argument("--secondary", default="targets_P117_AS.csv", help="Second-priority target list")
    parser.add_argument("--prev", default="prev_tgts.csv", help="Previously observed targets (list_targets.py CSV)")
    parser.add_argument("--date", help="Local evening date of the night, YYYY-MM-DD (default: tomorrow)")
    parser.add_argument("--min-alt", type=float, default=35.0, help="Minimum elevation (deg)")
    parser.add_argument("--sun-alt", type=float, default=-18.0, help="Sun altitude defining night (deg)")
    parser.add_argument("--min-hours", type=float, default=1.0,
                        help="Minimum continuous time above --min-alt, for each target and each group (h)")
    parser.add_argument("--max-sep", type=float, default=10.0, help="Maximum pairwise separation in a group (deg)")
    parser.add_argument("--max-size", type=int, default=3, help="Maximum targets per group")
    parser.add_argument("--prev-match", type=float, default=30.0, help="Match radius to previous targets (arcsec)")
    parser.add_argument("--out", help="Output basename (default: groups_<YYYYMMDD>)")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively")
    args = parser.parse_args()

    evening = (datetime.strptime(args.date, "%Y-%m-%d").date() if args.date
               else date.today() + timedelta(days=1))
    out = args.out or f"groups_{evening.strftime('%Y%m%d')}"

    # --- Targets: priority 1 first; drop priority-2 duplicates of priority-1 targets
    p1 = read_target_list(args.primary, 1)
    p2 = read_target_list(args.secondary, 2)
    p2 = [t for t, dup in zip(p2, _match_mask(p2, p1, 5.0)) if not dup]
    prev = read_prev(args.prev)
    allt = p1 + p2
    done = _match_mask(allt, prev, args.prev_match)
    todo = [t for t, d in zip(allt, done) if not d]

    # --- Observability
    times = night_grid(evening, args.sun_alt)
    min_steps = int(np.ceil(args.min_hours * 60 / TIME_STEP_MIN))
    up_all = altitudes(todo, times) > args.min_alt
    keep = [longest_run(row)[1] >= min_steps for row in up_all]
    targets = [t for t, k in zip(todo, keep) if k]
    up = up_all[keep]
    t_local = lambda i: times[i].to_datetime(timezone=LOCAL_TZ)
    print(f"Night of {evening} (local): dark from {t_local(0):%H:%M} to {t_local(-1):%H:%M} "
          f"{t_local(0):%Z} (Sun < {args.sun_alt:g} deg)")
    print(f"Targets: {len(p1)} priority 1, {len(p2)} priority 2 (after removing duplicates); "
          f"{int(done.sum())} already observed; {len(targets)} observable >{args.min_alt:g} deg "
          f"for >= {args.min_hours:g} h")
    print(f"Note: only elevation is modelled; {LARGE_CONFIG} delay-line/shadowing limits are not.")
    if not targets:
        raise SystemExit("Nothing to observe.")

    # --- Grouping
    groups = candidate_groups(targets, up, args.max_sep, min_steps, args.max_size)
    chosen = optimal_partition(len(targets), groups, args.max_sep)

    rows = []
    for members in chosen:
        members = sorted(members, key=lambda i: (targets[i]["priority"], targets[i]["coord"].ra.deg))
        start, length = longest_run(np.all(up[members], axis=0))
        sep = 0.0
        if len(members) > 1:
            sep = max(targets[a]["coord"].separation(targets[b]["coord"]).deg
                      for a, b in itertools.combinations(members, 2))
        n_p1 = sum(targets[i]["priority"] == 1 for i in members)
        rows.append(dict(members=members, priority=1 if n_p1 else 2, n_p1=n_p1,
                         start=start, stop=start + length - 1, max_sep=sep))
    rows.sort(key=lambda r: (r["priority"], r["start"], -len(r["members"])))
    for k, r in enumerate(rows, 1):
        r["group"] = k

    # --- CSV, one row per group
    import csv
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        head = ["targets", "group", "priority", "n_targets", "n_priority1", "max_sep_deg",
                "ut_start", "ut_end", "local_start", "local_end", "hours"]
        for k in range(1, args.max_size + 1):
            head += [f"name{k}", f"ra{k}", f"dec{k}", f"mag{k}", f"priority{k}"]
        w.writerow(head)
        for r in rows:
            ts, te = times[r["start"]], times[r["stop"]]
            line = [" ".join(obs_name(targets[i]["name"]) for i in r["members"]),
                    r["group"], r["priority"], len(r["members"]), r["n_p1"], f"{r['max_sep']:.2f}",
                    ts.datetime.strftime("%Y-%m-%d %H:%M"), te.datetime.strftime("%Y-%m-%d %H:%M"),
                    f"{t_local(r['start']):%H:%M}", f"{t_local(r['stop']):%H:%M}",
                    f"{(r['stop'] - r['start'] + 1) * TIME_STEP_MIN / 60:.2f}"]
            for k in range(args.max_size):
                if k < len(r["members"]):
                    t = targets[r["members"][k]]
                    line += [obs_name(t["name"]), t["ra"], t["dec"], t["mag"], t["priority"]]
                else:
                    line += [""] * 5
            w.writerow(line)

    # --- Summary
    print(f"\n{len(rows)} groups ({sum(len(r['members']) == 3 for r in rows)} triples, "
          f"{sum(len(r['members']) == 2 for r in rows)} pairs, "
          f"{sum(len(r['members']) == 1 for r in rows)} singles)")
    print(f"{'Grp':>3} {'Pri':>3} {'Local window':<12} {'Sep':>5}  Targets")
    for r in rows:
        names = ", ".join(targets[i]["name"] + ("*" if targets[i]["priority"] == 1 else "")
                          for i in r["members"])
        print(f"{r['group']:3d} {r['priority']:3d} {t_local(r['start']):%H:%M}-{t_local(r['stop']):%H:%M}  "
              f"{r['max_sep']:5.1f}  {names}")
    print("(* = priority 1)")

    # --- Plot, RA centred on the LST at the middle of the night
    lst_mid = times[len(times) // 2].sidereal_time("apparent", longitude=PARANAL.lon).hour
    plot_groups(rows, targets, prev, lst_mid,
                f"Heimdallr groups for night of {evening} (>{args.min_alt:g}° for ≥{args.min_hours:g} h, "
                f"≤{args.max_sep:g}° groups)", out + ".png")
    print(f"\nWrote {out}.csv and {out}.png")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
