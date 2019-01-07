import os

import denoising.image_utils as iu
from denoising.deep_image_prior import Denoiser
from os.path import join as jp


if __name__ == '__main__':

    steps = 100000000

    base_folder = jp('/', 'home', 'developer', 'photo_restoration', 'data')
    results_index = 6

    result_folder = jp(base_folder, 'results_{}'.format(str(results_index)))
    if os.path.isdir(result_folder):
        if len(os.listdir(result_folder)) > 0:
            raise ValueError('Folder "{}" already exists'.format(result_folder))
    else:
        os.mkdir(result_folder)

    crops_img_index = 49
    crop_index = 2
    use_crops = False

    mask_filename = 'IMG-0.png'
    image_filename = 'test_noisy.png'

    if use_crops:
        base_folder = jp(base_folder, 'crops_{}'.format(str(crop_index)))
        mask_path = jp(base_folder, 'mask', 'IMG-{}.png'.format(str(crops_img_index)))
        noisy_path = jp(base_folder, 'source', 'IMG-{}.png'.format(str(crops_img_index)))
    else:
        mask_path = jp(base_folder, mask_filename)
        noisy_path = jp(base_folder, image_filename)

    noisy = iu.file_to_tensor(noisy_path)
    mask = iu.file_to_tensor(mask_path)

    Denoiser(result_folder, num_steps=steps).denoise(mask, noisy)
