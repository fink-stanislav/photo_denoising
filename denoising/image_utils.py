
from PIL import Image
import torchvision
import torch
import numpy as np
import math


def _calc_valid_sizes(last_valid_size=16):
    valid_sizes = [64 * i for i in range(1, last_valid_size + 1)]
    return valid_sizes

def calc_preferrable_size(w, h):
    size = max([w, h])
    valid_sizes = _calc_valid_sizes()
    diffs = [abs(size - valid_size) for valid_size in valid_sizes]
    index_min = np.argmin(diffs)
    size = valid_sizes[index_min]
    return size

def resize(pil, w, h):
    resized = pil.resize((w, h), Image.ANTIALIAS)
    return resized

def psnr(result, original):
    target_data = np.array(result, dtype=np.float64)
    ref_data = np.array(original, dtype=np.float64)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))

    if rmse == 0.0:
        return math.inf

    return 20 * math.log10(255 / rmse)

def file_to_tensor(filepath, use_cuda=True):
    """
    Accepts a file path to a image, return a torch tensor
    """
    pil = Image.open(filepath)
    return pil_to_tensor(pil, use_cuda=use_cuda)

def pil_to_tensor(pil, use_cuda=True):
    """
    Accepts a PIL image, return a torch tensor
    """
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    if use_cuda:
        tensor = transform(pil).cuda()
    else:
        tensor = transform(pil)
    return tensor.view([1]+list(tensor.shape))

def tensor_to_file(tensor, filename, use_cuda=True):
    """
    Accepts a torch tensor, convert it to an image at a certain path
    """
    pil = tensor_to_pil(tensor, use_cuda=use_cuda)
    pil.save(filename)

def tensor_to_pil(tensor, use_cuda=True):
    """
    Accepts a torch tensor, convert it to a PIL image
    """
    tensor = tensor.view(tensor.shape[1:])
    if use_cuda:
        tensor = tensor.cpu()
    transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
    pil = transform(tensor)
    return pil

def generate_noise(tensor, prop=0.5, use_cuda=True):
    """
    Generates noise - zeroes out a random proportion of pixels from an image tensor.
    """
    if use_cuda:
        mask = torch.rand([1]+[1] + list(tensor.shape[2:])).cuda()
    else:
        mask = torch.rand([1]+[1] + list(tensor.shape[2:]))
    mask[mask < prop] = 0
    mask[mask != 0] = 1
    mask = mask.repeat(1, tensor.shape[1], 1, 1)
    deconstructed = tensor * mask
    return mask, deconstructed
