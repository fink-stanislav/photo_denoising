
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from PIL import Image
import skimage
from skimage.measure.simple_metrics import compare_psnr
import numpy as np

class NoiseGenerator(object):

    def __init__(self):
        self.params = {
            'gaussian': {'var': 0.01, 'mode': 'gaussian', 'mean': 0}, 
            'poisson': {'mode': 'poisson'},
            'salt': {'mode': 'salt', 'amount': 0.5},
            'pepper': {'mode': 'pepper', 'amount': 0.5},
            's&p': {'mode': 's&p', 'amount': 0.5, 'salt_vs_pepper': 0.5}, 
            'speckle': {'var': 0.1, 'mode': 'speckle', 'mean': 0}
        }

    def _add_noise(self, params):
        noisy_img = skimage.util.random_noise(**params)
        mask = params['image'] - noisy_img
        return mask, noisy_img

    def _prepare_params(self, img, name):
        params = self.params[name]
        np_image = np.asarray(img, dtype="float")
        params['image'] = np_image / 255
        return params

    def gaussian(self, img):
        params = self._prepare_params(img, 'gaussian')
        mask, noisy_img = self._add_noise(params)
        return mask, noisy_img

    def poisson(self, img):
        params = self._prepare_params(img, 'poisson')
        mask, noisy_img = self._add_noise(params)
        return mask, noisy_img
    
    def salt(self, img):
        params = self._prepare_params(img, 'salt')
        mask, noisy_img = self._add_noise(params)
        mask[mask != 0] = 1
        return mask, noisy_img

    def pepper(self, img):
        params = self._prepare_params(img, 'pepper')
        mask, noisy_img = self._add_noise(params)
        mask[mask != 1] = 0
        return mask, noisy_img

class Denoiser(object):
    
    def __init__(self, output_dir, use_cuda=True):
        self.use_cuda = use_cuda
        self.output_dir = output_dir
        self.sigma = 1./30
        self.num_steps = 25001
        self.save_frequency = 100

    def _np_to_tensor(self, np_data):
        np_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        if self.use_cuda:
            tensor = np_to_tensor(np_data).cuda()
        else:
            tensor = np_to_tensor(np_data)
        return tensor.view([1] + list(tensor.shape))

    #accept a file path to a jpg, return a torch tensor
    def jpg_to_tensor(self, filepath):
        pil = Image.open(filepath)
        pil_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        if self.use_cuda:
            tensor = pil_to_tensor(pil).cuda()
        else:
            tensor = pil_to_tensor(pil)
        return tensor.view([1]+list(tensor.shape))
    
    def jpg_to_tensor_np(self, filepath):
        pil = Image.open(filepath)
        data = np.asarray(pil, dtype="float")
        data = data / 255
        return self._np_to_tensor(data)

    #accept a torch tensor, convert it to a jpg at a certain path
    def tensor_to_jpg(self, tensor, filename):
        tensor = tensor.view(tensor.shape[1:])
        if self.use_cuda:
            tensor = tensor.cpu()
        tensor_to_pil = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
        pil = tensor_to_pil(tensor)
        pil.save(filename)
    
    #function which zeros out a random proportion of pixels from an image tensor.
    def zero_out_pixels(self, tensor, prop=0.5):
        if self.use_cuda:
            mask = torch.rand([1]+[1] + list(tensor.shape[2:])).cuda()
        else:
            mask = torch.rand([1]+[1] + list(tensor.shape[2:]))
        mask[mask<prop] = 0
        mask[mask!=0] = 1
        mask = mask.repeat(1,3,1,1)
        deconstructed = tensor * mask
        return mask, deconstructed
    
    def denoise(self, mask, deconstructed):
        deconstructed = self._np_to_tensor(deconstructed).float()
        self.tensor_to_jpg(deconstructed, '../deconstructed.jpg')
        mask = self._np_to_tensor(mask).float()
        self.tensor_to_jpg(mask, '../mask.jpg')
        #convert the image and mask to variables.
        mask = Variable(mask)
        deconstructed = Variable(deconstructed)
    
        #input of the network is noise
        if self.use_cuda:
            noise = Variable(torch.randn(deconstructed.shape).cuda())
        else:
            noise = Variable(torch.randn(deconstructed.shape))
    
        #initialise the network with the chosen architecture
        net = PixelShuffleHourglass()
        #net = UNet(3, 3)
    
        #bind the network to the gpu if cuda is enabled
        if self.use_cuda:
            net.cuda()
        #network optimizer set up
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    
        #dummy index to provide names to output files
        save_img_ind = 0
        for step in range(self.num_steps):
            #get the network output
            output = net(noise)
            #we are only concerned with the output where we have the image available.
            masked_output = output * mask
            # calculate the l2_loss over the masked output and take an optimizer step
            optimizer.zero_grad()
            loss = torch.sum((masked_output - deconstructed)**2)
            #loss = torch.sum((output - deconstructed)**2)
            loss.backward()
            optimizer.step()
            #every save_frequency steps, save a jpg
            print('At step {}, loss is {}'.format(step, loss.data.cpu()))
            if step % self.save_frequency == 0:
                output_path = '{}/output_{}.jpg'.format(self.output_dir, save_img_ind)
                self.tensor_to_jpg(output.data, output_path)
                save_img_ind += 1
            if self.use_cuda:
                noise.data += self.sigma * torch.randn(noise.shape).cuda()
            else:
                noise.data += self.sigma * torch.randn(noise.shape)

        #clean up any mess we're leaving on the gpu
        if self.use_cuda:
            torch.cuda.empty_cache()

#define an encoder decoder network with pixel shuffle upsampling
class PixelShuffleHourglass(nn.Module):
    def __init__(self):
        super(PixelShuffleHourglass, self).__init__()
        self.d_conv_1 = nn.Conv2d(3, 8, 5, stride=2, padding=2)
        self.d_bn_1 = nn.BatchNorm2d(8)

        self.d_conv_2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.d_bn_2 = nn.BatchNorm2d(16)

        self.d_conv_3 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.d_bn_3 = nn.BatchNorm2d(32)
        self.s_conv_3 = nn.Conv2d(32, 4, 5, stride=1, padding=2)

        self.d_conv_4 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.d_bn_4 = nn.BatchNorm2d(64)
        self.s_conv_4 = nn.Conv2d(64, 4, 5, stride=1, padding=2)

        self.d_conv_5 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.d_bn_5 = nn.BatchNorm2d(128)
        self.s_conv_5 = nn.Conv2d(128, 4, 5, stride=1, padding=2)

        self.d_conv_6 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.d_bn_6 = nn.BatchNorm2d(256)
        self.s_conv_6 = nn.Conv2d(256, 4, 5, stride=1, padding=2)
        
        self.d_conv_7 = nn.Conv2d(256, 512, 5, stride=2, padding=2)
        self.d_bn_7 = nn.BatchNorm2d(512)
        
        self.u_conv_6 = nn.Conv2d(132, 256, 5, stride=1, padding=2)
        self.u_bn_6 = nn.BatchNorm2d(256)

        self.u_conv_5 = nn.Conv2d(68, 128, 5, stride=1, padding=2)
        self.u_bn_5 = nn.BatchNorm2d(128)

        self.u_conv_4 = nn.Conv2d(36, 64, 5, stride=1, padding=2)
        self.u_bn_4 = nn.BatchNorm2d(64)

        self.u_conv_3 = nn.Conv2d(20, 32, 5, stride=1, padding=2)
        self.u_bn_3 = nn.BatchNorm2d(32)

        self.u_conv_2 = nn.Conv2d(8, 16, 5, stride=1, padding=2)
        self.u_bn_2 = nn.BatchNorm2d(16)

        self.u_conv_1 = nn.Conv2d(4, 16, 5, stride=1, padding=2)
        self.u_bn_1 = nn.BatchNorm2d(16)

        self.out_conv = nn.Conv2d(4, 3, 5, stride=1, padding=2)
        self.out_bn = nn.BatchNorm2d(3)

    def forward(self, noise):
        down_1 = self.d_conv_1(noise)
        down_1 = self.d_bn_1(down_1)
        down_1 = F.leaky_relu(down_1)
        
        down_2 = self.d_conv_2(down_1)
        down_2 = self.d_bn_2(down_2)
        down_2 = F.leaky_relu(down_2)

        down_3 = self.d_conv_3(down_2)
        down_3 = self.d_bn_3(down_3)
        down_3 = F.leaky_relu(down_3)
        skip_3 = self.s_conv_3(down_3)

        down_4 = self.d_conv_4(down_3)
        down_4 = self.d_bn_4(down_4)
        down_4 = F.leaky_relu(down_4)
        skip_4 = self.s_conv_4(down_4)

        down_5 = self.d_conv_5(down_4)
        down_5 = self.d_bn_5(down_5)
        down_5 = F.leaky_relu(down_5)
        skip_5 = self.s_conv_5(down_5)

        down_6 = self.d_conv_6(down_5)
        down_6 = self.d_bn_6(down_6)
        down_6 = F.leaky_relu(down_6)
        skip_6 = self.s_conv_6(down_6)
        
        down_7 = self.d_conv_7(down_6)
        down_7 = self.d_bn_7(down_7)
        down_7 = F.leaky_relu(down_7)
        
        up_6 = F.pixel_shuffle(down_7, 2)
        up_6 = torch.cat([up_6, skip_6], 1)
        up_6 = self.u_conv_6(up_6)
        up_6 = self.u_bn_6(up_6)
        up_6 = F.leaky_relu(up_6)

        up_5 = F.pixel_shuffle(up_6, 2)
        up_5 = torch.cat([up_5, skip_5], 1)
        up_5 = self.u_conv_5(up_5)
        up_5 = self.u_bn_5(up_5)
        up_5 = F.leaky_relu(up_5)

        up_4 = F.pixel_shuffle(up_5, 2)
        up_4 = torch.cat([up_4, skip_4], 1)
        up_4 = self.u_conv_4(up_4)
        up_4 = self.u_bn_4(up_4)
        up_4 = F.leaky_relu(up_4)

        up_3 = F.pixel_shuffle(up_4, 2)
        up_3 = torch.cat([up_3, skip_3], 1)
        up_3 = self.u_conv_3(up_3)
        up_3 = self.u_bn_3(up_3)
        up_3 = F.leaky_relu(up_3)

        up_2 = F.pixel_shuffle(up_3, 2)
        up_2 = self.u_conv_2(up_2)
        up_2 = self.u_bn_2(up_2)
        up_2 = F.leaky_relu(up_2)

        up_1 = F.pixel_shuffle(up_2, 2)
        up_1 = self.u_conv_1(up_1)
        up_1 = self.u_bn_1(up_1)
        up_1 = F.leaky_relu(up_1)

        out = F.pixel_shuffle(up_1, 2)
        out = self.out_conv(out)
        out = self.out_bn(out)
        out = F.sigmoid(out)
        return out


if __name__=='__main__':
    denoiser = Denoiser('output')
    #mask, deconstructed = NoiseGenerator().gaussian(Image.open('../bunny_512.jpg'))
    mask, deconstructed = NoiseGenerator().salt(Image.open('../bunny_512.jpg'))
    denoiser.denoise(mask, deconstructed)
    
