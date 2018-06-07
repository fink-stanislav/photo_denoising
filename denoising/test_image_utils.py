
import unittest

import denoising.image_utils as iu


class TestImageUtils(unittest.TestCase):


    def test_calc_valid_sizes(self):
        actual = iu._calc_valid_sizes()
        expected = [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024]
        self.assertListEqual(actual, expected)

    def test_calc_preferrable_size(self):
        size = iu.calc_preferrable_size(622, 415)
        self.assertEqual(size, 640)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()