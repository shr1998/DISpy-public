import unittest

from dispersion_params import resolve_dispersion_params


class DispersionParameterPriorityTests(unittest.TestCase):
    def test_cc_file_geometry_and_sampling_take_priority(self):
        dispersion = {
            "dr": 5.0,
            "step": 10,
            "cc_len": 8,
            "dt": 0.02,
            "time_range": 12,
            "fmax": 20,
        }
        cc_config = {
            "dr": 8.0,
            "step": 20,
            "cc_len": 10,
            "samplerate_in_use": 100,
            "time_range": 5,
        }
        cc_file = {
            "dr": 10.0,
            "step": 25,
            "cc_len": 14,
            "samplerate_in_use": 200,
            "time_range": 3,
        }

        resolved = resolve_dispersion_params(dispersion, cc_config, cc_file)

        self.assertEqual(resolved["dr"], 10.0)
        self.assertEqual(resolved["step"], 25)
        self.assertEqual(resolved["cc_len"], 14)
        self.assertAlmostEqual(resolved["dt"], 0.005)
        self.assertEqual(resolved["time_range"], 12)
        self.assertEqual(resolved["fmax"], 20)

    def test_current_cc_config_is_the_fallback_without_file_metadata(self):
        resolved = resolve_dispersion_params(
            {"dr": 5.0, "dt": 0.02, "time_range": 10},
            {"dr": 7.5, "delta": 0.01},
            {},
        )

        self.assertEqual(resolved["dr"], 7.5)
        self.assertAlmostEqual(resolved["dt"], 0.01)
        self.assertEqual(resolved["time_range"], 10)

    def test_h5_metadata_aliases_are_recognized(self):
        resolved = resolve_dispersion_params(
            {"dr": 5.0, "dt": 0.02, "step": 25, "c_range": 100},
            {},
            {
                "channel_spacing_m": 8.179959,
                "delta_s": 0.04,
                "source_step_channels": 100,
                "c_range_channels": 1220,
            },
        )

        self.assertAlmostEqual(resolved["dr"], 8.179959)
        self.assertAlmostEqual(resolved["dt"], 0.04)
        self.assertEqual(resolved["step"], 100)
        self.assertEqual(resolved["c_range"], 1220)


if __name__ == "__main__":
    unittest.main()
