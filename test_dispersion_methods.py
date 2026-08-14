import unittest

import numpy as np

import disp_cal


class DispersionMethodTests(unittest.TestCase):
    def setUp(self):
        dt = 0.02
        dr = 5.0
        ntime = 400
        center = ntime // 2
        cc = np.zeros((2, 18, ntime), dtype=np.float32)
        for source in range(cc.shape[0]):
            for trace in range(1, cc.shape[1]):
                offset = trace * dr
                arrival = center + int(round(offset / 300.0 / dt))
                if arrival < ntime:
                    cc[source, trace, arrival] = 1.0
        self.cc = cc
        self.params = {
            'range': [0, 1000],
            'step': 25,
            'fmin': 1.0,
            'fmax': 8.0,
            'vmin': 150.0,
            'vmax': 500.0,
            'dv': 25.0,
            'dt': dt,
            'dr': dr,
            'time_range': 4.0,
            'Part': 'causal',
        }

    def _run(self, method):
        received = []
        method(self.params, self.cc, lambda value: None, received.append, lambda value: None)
        self.assertEqual(len(received), 1)
        image, extent = received[0]
        self.assertEqual(image.shape[0], 2)
        self.assertEqual(image.shape[1], 15)
        self.assertTrue(np.isfinite(image).all())
        self.assertGreater(image.shape[2], 1)
        self.assertEqual(extent[2:], [150.0, 500.0])

    def test_cc_fj(self):
        self._run(disp_cal.cc_fj)

    def test_single_source_dimension_is_preserved(self):
        cube = disp_cal._ensure_cc_cube(self.cc[:1])
        self.assertEqual(cube.shape, (1, 18, 400))


if __name__ == '__main__':
    unittest.main()
