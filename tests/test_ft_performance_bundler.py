import datetime
import pathlib
import sys
import tempfile
import unittest

import numpy as onp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from heimdallrdr.FTPerformanceBundler import FTPerformanceBundler


class FTPerformanceBundlerTests(unittest.TestCase):
    START = datetime.datetime(2026, 10, 10, 13, 45, tzinfo=datetime.timezone.utc)
    GROUP_NAMES = "gd_snr pd_snr gd_bl pd_tel gd_tel dm_piston"

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_log(self, offset, measurements, trailing=(), named_trailing=()):
        timestamp = self.START + datetime.timedelta(seconds=offset)
        name = f"ft_performance_{timestamp:%Y%m%dT%H%M%S}.log"
        header = f"# timestamp {self.GROUP_NAMES}"
        if named_trailing:
            header += " " + " ".join(named_trailing)
        header += " (values are space-separated)\n"
        values = [
            f"{timestamp.timestamp():.4f}",
            *map(str, measurements),
            *map(str, trailing),
        ]
        (self.path / name).write_text(header + " ".join(values) + "\n")

    def test_legacy_and_new_logs_keep_measurements_and_counters_aligned(self):
        self._write_log(0, range(1, 31))
        self._write_log(1, range(101, 131), trailing=(901,), named_trailing=("cnt",))
        newer_measurements = list(range(201, 231))
        newer_measurements[1] = "nan"
        self._write_log(
            2,
            newer_measurements,
            trailing=(902, 42, 4),
            named_trailing=("cnt", "seq", "servo_mode"),
        )

        bundler = FTPerformanceBundler(self.path)
        df, coverage = bundler.get_performance_between(
            self.START, self.START + datetime.timedelta(seconds=3)
        )

        self.assertEqual(len(df), 3)
        self.assertEqual(coverage, 1.0)
        self.assertEqual(df["gd_snr_1"].tolist(), [1, 101, 201])
        self.assertEqual(df["gd_bl_6"].tolist(), [18, 118, 218])
        self.assertEqual(df["dm_piston_4"].tolist(), [30, 130, 230])
        self.assertTrue(onp.isnan(df.loc[0, "cnt"]))
        self.assertEqual(df.loc[1, "cnt"], 901)
        self.assertEqual(df.loc[2, "cnt"], 902)
        self.assertEqual(df.loc[2, "seq"], 42)
        self.assertEqual(df.loc[2, "servo_mode"], 4)

        groups = {
            "gd_snr": range(1, 7),
            "pd_snr": range(7, 13),
            "gd_bl": range(13, 19),
            "pd_tel": range(19, 23),
            "gd_tel": range(23, 27),
            "dm_piston": range(27, 31),
        }
        for name, expected in groups.items():
            with self.subTest(name=name):
                array = bundler.measurement_array(df, name)
                self.assertEqual(array.shape[1], len(expected))
                onp.testing.assert_allclose(array[0], expected)

        gd_snr = bundler.measurement_array(df, "gd_snr")
        self.assertEqual(gd_snr.shape, (3, 6))
        self.assertTrue(onp.isnan(gd_snr[2, 1]))
        dm_piston = bundler.measurement_array(df, "dm_piston")
        self.assertEqual(dm_piston.shape, (3, 4))
        onp.testing.assert_allclose(dm_piston[1], range(127, 131))

    def test_missing_header_counter_names_and_unknown_extra(self):
        self._write_log(0, range(1, 31), trailing=(901, 42, 4, 77))

        bundler = FTPerformanceBundler(self.path)
        df, _ = bundler.get_performance_between(
            self.START, self.START + datetime.timedelta(seconds=1)
        )

        self.assertEqual(df.loc[0, "cnt"], 901)
        self.assertEqual(df.loc[0, "seq"], 42)
        self.assertEqual(df.loc[0, "servo_mode"], 4)
        self.assertEqual(df.loc[0, "aux_01"], 77)

    def test_empty_measurement_array_has_group_width(self):
        bundler = FTPerformanceBundler(self.path)
        df, coverage = bundler.get_performance_between(
            self.START, self.START + datetime.timedelta(seconds=1)
        )

        self.assertEqual(coverage, 0.0)
        self.assertEqual(bundler.measurement_array(df, "gd_snr").shape, (0, 6))
        self.assertEqual(bundler.measurement_array(df, "dm_piston").shape, (0, 4))
        with self.assertRaises(ValueError):
            bundler.measurement_array(df, "unknown")


if __name__ == "__main__":
    unittest.main()
