
import unittest

import denoising.image_utils as iu
from PIL import Image
import math


class TestImageUtils(unittest.TestCase):

    def test_calc_valid_sizes(self):
        actual = iu._calc_valid_sizes()
        expected = [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024]
        self.assertListEqual(actual, expected)

    def test_calc_preferrable_size(self):
        size = iu.calc_preferrable_size(622, 415)
        self.assertEqual(size, 640)

    def test_psnr_identical(self):
        img1 = Image.open('test_images/barbara.png')
        img2 = Image.open('test_images/barbara_same.png')
        self.assertEqual(iu.psnr(img2, img1), math.inf)

    def test_psnr_noisy_denoised(self):
        original = Image.open('test_images/barbara.png')
        noisy = Image.open('test_images/barbara_noisy.png')
        denoised = Image.open('test_images/barbara_denoised.png')
        psnr1 = iu.psnr(original, noisy)
        psnr2 = iu.psnr(original, denoised)
        self.assertTrue(psnr1 < psnr2)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()