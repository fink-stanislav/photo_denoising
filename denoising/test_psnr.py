
import unittest
import math
import numpy as np
import cv2


class TestPsnr(unittest.TestCase):

    def psnr(self, target, r):
        target_data = np.array(target, dtype=np.float64)
        ref_data = np.array(r, dtype=np.float64)
    
        diff = ref_data - target_data
        print(diff.shape)
        diff = diff.flatten('C')
    
        rmse = math.sqrt(np.mean(diff ** 2.))
    
        return 20 * math.log10(255 / rmse)

    def calc_psnr(self, p1, p2):
        i1 = cv2.imread(p1)
        i2 = cv2.imread(p2)

        i1= cv2.cvtColor(i1, cv2.COLOR_BGR2YCrCb)
        i2= cv2.cvtColor(i2, cv2.COLOR_BGR2YCrCb)
        
        return self.psnr(i1[:,:,0], i2[:,:,0])

    def test_denoising(self):
        print(self.calc_psnr('../bunny_512.jpg', '../deconstructed.jpg'))
        print(self.calc_psnr('../bunny_512.jpg', '../output_100.jpg'))
        print(self.calc_psnr('../bunny_512.jpg', '../output_40.jpg'))
        print(self.calc_psnr('../bunny_512.jpg', '../output_0.jpg'))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()