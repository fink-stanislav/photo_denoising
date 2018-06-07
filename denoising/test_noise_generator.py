
import unittest

from denoising.deep_image_prior import NoiseGenerator, Denoiser
from PIL import Image
from skimage.util.dtype import img_as_float
import numpy as np
from skimage.util.noise import random_noise


class TestNoiseGenerator(unittest.TestCase):

    def test_gaussian(self):
        ng = NoiseGenerator()
        pil = Image.open('../bunny_512.jpg')
        ng.gaussian(pil)

    def test_noise_generation(self):
        pil = Image.open('../bunny_512.jpg')
        np_image = img_as_float(np.asarray(pil))
        noisy = random_noise(np_image, mode='salt', amount=0.5)
        noisy *= 255
        Image.fromarray(np.uint8(noisy)).save('test_results/noisy_salt.png')
    
    def test_create_mask(self):
        pil = Image.open('../bunny_512.jpg')
        np_image = img_as_float(np.asarray(pil))
        noisy = random_noise(np_image, mode='salt', amount=0.01)
        mask = noisy
        
        for row in mask:
            for i, pixel in enumerate(row):
                if 1.0 in pixel:
                    row[i] = np.asarray([1.0, 1.0, 1.0])
                else:
                    row[i] = np.asarray([0.0, 0.0, 0.0])
        
        mask *= 255
        Image.fromarray(np.uint8(mask)).save('test_results/mask_salt.png')

    def test_zeroing(self):
        d = Denoiser(None)
        truth = d.jpg_to_tensor('../results/bunny_512.jpg')
        mask, deconstructed = d.generate_noise(truth)
        print(mask)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()