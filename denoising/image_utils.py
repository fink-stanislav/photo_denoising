
from PIL import Image
import torchvision
import torch

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
    mask = mask.repeat(1, 3, 1, 1)
    deconstructed = tensor * mask
    return mask, deconstructed
