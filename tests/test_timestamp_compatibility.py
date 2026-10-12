import contextlib
import datetime
import io
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as onp
from astropy.io import fits

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heimdallrdr.DataBundler import DataBundler
from heimdallrdr.FTPerformanceBundler import FTPerformanceBundler
from heimdallrdr.FtSettingsBundler import FTSettingsBundler
from heimdallrdr.TelemBundler import load_ttmet
from heimdallrdr._timestamps import (
    parse_camera_timestamp,
    parse_time_seconds_and_str,
    utc_timestamp,
)


UTC = datetime.timezone.utc


class TimestampParsingTests(unittest.TestCase):
    def test_camera_variants_and_precision(self):
        day = datetime.date(2026, 10, 10)
        for clock in ("12:34:56", "12-34-56", "12_34_56", "123456"):
            for fraction, microseconds in (
                ("", 0),
                (".1", 100000),
                (".123", 123000),
                (".123456", 123456),
                (".123456789", 123456),
            ):
                for prefix in ("baldr1_T", "hei_k1_2026-10-10T"):
                    name = prefix + clock + fraction + ".fits"
                    with self.subTest(name=name):
                        self.assertEqual(
                            parse_camera_timestamp(name, day),
                            datetime.datetime(2026, 10, 10, 12, 34, 56, microseconds, UTC),
                        )
                        seconds, printable = parse_time_seconds_and_str(name)
                        self.assertAlmostEqual(seconds, 45296 + microseconds / 1e6)
                        self.assertEqual(printable, "12:34:56" + fraction[:7])

    def test_invalid_names_do_not_parse_partial_timestamps(self):
        for name in (
            "baldr1_T25:00:00.fits",
            "baldr1_T12:60:00.fits",
            "baldr1_T12:00:60.fits",
            "baldr1_T12:34-56.fits",
            "baldr1_T1234567.fits",
            "baldr1_T12:34:56.invalid.fits",
            "baldr1_2026-02-30T12:00:00.fits",
            "baldr1_flat.fits",
        ):
            with self.subTest(name=name):
                self.assertIsNone(parse_camera_timestamp(name, datetime.date(2026, 10, 10)))

    def test_explicit_date_wins_over_folder(self):
        self.assertEqual(
            parse_camera_timestamp("baldr1_2026-10-10T00-00-01.fits", datetime.date(2026, 10, 9)),
            datetime.datetime(2026, 10, 10, 0, 0, 1, tzinfo=UTC),
        )


class ReaderUtcTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(time, "tzset"), "requires process timezone control")
    def test_tt_selection_matches_other_readers_on_non_utc_hosts(self):
        start = datetime.datetime(2026, 10, 10, 12, tzinfo=UTC)
        epoch = start.timestamp()
        local_zone = datetime.timezone(datetime.timedelta(hours=-3))
        bounds = (
            (start.replace(tzinfo=None), (start + datetime.timedelta(seconds=0.5)).replace(tzinfo=None)),
            (start, start + datetime.timedelta(seconds=0.5)),
            (start.astimezone(local_zone), (start + datetime.timedelta(seconds=0.5)).astimezone(local_zone)),
            (epoch, epoch + 0.5),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "ttmet.log"
            path.write_text("".join(f"{epoch + offset} {offset} 0 0 0\n" for offset in (-1, 0, 1)))
            for zone in ("UTC0", "UTC+3", "UTC-10"):
                try:
                    with mock.patch.dict(os.environ, {"TZ": zone}):
                        time.tzset()
                        for lower, upper in bounds:
                            with self.subTest(zone=zone, bounds=type(lower).__name__):
                                with contextlib.redirect_stderr(io.StringIO()):
                                    rows = load_ttmet(path, lower, upper)
                                self.assertEqual(rows["tx"].tolist(), [0.0])
                                self.assertEqual(rows["time"].tolist(), [start])
                                self.assertEqual(str(rows["time"].dt.tz), "UTC")
                                self.assertEqual(FTPerformanceBundler._coerce_utc_datetime(lower), start)
                                self.assertEqual(FTSettingsBundler._coerce_utc_datetime(lower), start)
                                self.assertEqual(DataBundler(directory)._coerce_utc_naive(lower), start.replace(tzinfo=None))
                finally:
                    time.tzset()

    def test_numeric_bounds_retain_their_precision(self):
        value = 1775258894.1234567
        self.assertEqual(utc_timestamp(value), value)
        self.assertEqual(utc_timestamp(float("inf")), float("inf"))


class CameraSelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = pathlib.Path(self.directory.name)

    def write_cube(self, folder, name, values, fps=2):
        path = self.root / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        hdu = fits.CompImageHDU(onp.asarray(values, dtype=onp.int16).reshape(-1, 1, 1))
        hdu.header["FPS"] = fps
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)

    def load(self, start, end):
        with contextlib.redirect_stderr(io.StringIO()):
            return DataBundler(self.root).load_between("baldr1", start, end)

    def test_mixed_formats_select_frames_in_chronological_order(self):
        names = (
            "baldr1_T12_00_00.000.fits",
            "baldr1_T12-00-01.000.fits",
            "baldr1_T12:00:02.000.fits",
            "baldr1_T120003.fits",
            "baldr1_2026-10-10T12-00-04.fits",
        )
        for index, name in enumerate(names):
            self.write_cube("20261010", name, [2 * index, 2 * index + 1])
        start = datetime.datetime(2026, 10, 10, 12, 0, 0, 500000, UTC)
        end = start + datetime.timedelta(seconds=4)
        for lower, upper in ((start, end), (start.replace(tzinfo=None), end.replace(tzinfo=None))):
            with self.subTest(aware=lower.tzinfo is not None):
                used, times, frames = self.load(lower, upper)
                onp.testing.assert_array_equal(frames[:, 0, 0], onp.arange(1, 9))
                expected = onp.datetime64("2026-10-10T12:00:00.500000", "us") + onp.arange(8) * onp.timedelta64(500000, "us")
                onp.testing.assert_array_equal(times, expected)
                self.assertEqual(used, sorted(used))
                self.assertTrue(all(stamp.tzinfo is None for stamp in used))
                self.assertEqual(times.dtype, onp.dtype("datetime64[us]"))

    def test_previous_day_cube_can_overlap_utc_midnight(self):
        self.write_cube("20261009", "baldr1_T23:59:59.500.fits", [10, 11, 12, 13])
        start = datetime.datetime(2026, 10, 10, tzinfo=UTC)
        _, times, frames = self.load(start, start + datetime.timedelta(seconds=1))
        onp.testing.assert_array_equal(frames[:, 0, 0], [11, 12])
        onp.testing.assert_array_equal(times, onp.array(["2026-10-10T00:00:00", "2026-10-10T00:00:00.5"], dtype="datetime64[us]"))

    def test_dated_file_uses_embedded_date_in_previous_day_folder(self):
        self.write_cube("20261009", "baldr1_2026-10-10T00-00-00.000.fits", [20, 21])
        start = datetime.datetime(2026, 10, 10, tzinfo=UTC)
        used, _, frames = self.load(start, start + datetime.timedelta(seconds=1))
        self.assertEqual(used, [start.replace(tzinfo=None)])
        onp.testing.assert_array_equal(frames[:, 0, 0], [20, 21])


class StandaloneScriptTests(unittest.TestCase):
    def test_find_blocks_runs_without_an_installed_package(self):
        with tempfile.TemporaryDirectory() as directory:
            locked = pathlib.Path(directory) / "locked"
            locked.mkdir()
            for name in (
                "hei_k2_T12:00:00.123.fits",
                "hei_k2_T12_00_01.123456789.fits",
                "hei_k2_2026-10-10T12-00-30.fits",
            ):
                (locked / name).touch()
            result = subprocess.run([sys.executable, "-S", str(ROOT / "find_blocks.py")], cwd=directory, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.splitlines(), ["12:00:00.123 12:00:01.123456", "12:00:30 12:00:30"])

    def test_compress_help_keeps_stdlib_only_standalone_workflow(self):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "compress_blocks.py"), "--help"], cwd=tempfile.gettempdir(), capture_output=True, text=True, check=True)
        self.assertIn("--time_start", result.stdout)
        self.assertIn("HHMMSS", result.stdout)


if __name__ == "__main__":
    unittest.main()
