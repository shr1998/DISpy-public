import unittest

import numpy as np

from dispersion_centerline import extract_dispersion_centerlines, order_centerlines_by_min_frequency


class DispersionCenterlineTests(unittest.TestCase):
    def test_orders_curves_by_lower_frequency_limit(self):
        curves = [
            np.array([[17.0, 300.0], [20.0, 295.0]]),
            np.array([[6.0, 580.0], [10.0, 400.0]]),
            np.array([[2.0, 640.0], [5.0, 350.0]]),
            np.array([[31.0, 510.0], [36.0, 440.0]]),
        ]
        self.assertEqual(order_centerlines_by_min_frequency(curves), [2, 1, 0, 3])

    def test_extracts_two_curved_bands(self):
        height, width = 180, 240
        x = np.arange(width)
        first = 42 + 18 * np.exp(-x / 75.0)
        second = 98 + 25 * np.exp(-x / 90.0)
        rows = np.arange(height)[:, None]
        image = np.exp(-0.5 * ((rows - first[None, :]) / 5.0) ** 2)
        image += 0.9 * np.exp(-0.5 * ((rows - second[None, :]) / 6.0) ** 2)

        curves, mask = extract_dispersion_centerlines(
            image,
            extent=(2.0, 40.0, 150.0, 800.0),
            threshold=0.35,
            min_area=80,
            min_width=30,
            sample_step=1,
        )

        self.assertEqual(len(curves), 2)
        self.assertGreater(np.count_nonzero(mask), 0)
        curve_x0 = (curves[0][:, 0] - 2.0) * (width - 1) / 38.0
        curve_x1 = (curves[1][:, 0] - 2.0) * (width - 1) / 38.0
        expected_first = 150.0 + np.interp(curve_x0, x, first) * 650.0 / (height - 1)
        expected_second = 150.0 + np.interp(curve_x1, x, second) * 650.0 / (height - 1)
        self.assertLess(np.mean(np.abs(curves[0][:, 1] - expected_first)), 8.0)
        self.assertLess(np.mean(np.abs(curves[1][:, 1] - expected_second)), 8.0)

    def test_removes_small_fragments(self):
        image = np.zeros((80, 100), dtype=float)
        image[20:28, 10:90] = 1.0
        image[60:62, 4:7] = 1.0
        curves, _ = extract_dispersion_centerlines(
            image,
            extent=(2.0, 40.0, 150.0, 800.0),
            threshold=0.5,
            min_area=30,
            min_width=10,
        )
        self.assertEqual(len(curves), 1)


if __name__ == "__main__":
    unittest.main()
