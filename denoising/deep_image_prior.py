
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Denoiser(object):

    def __init__(self, num_steps=5001, min_loss=500, use_cuda=True):
        self.use_cuda = use_cuda
        self.sigma = 1./30
        self.num_steps = num_steps
        self.min_loss = min_loss

    def denoise(self, mask, deconstructed):
        mask = Variable(mask)
        deconstructed = Variable(deconstructed)

        #input of the network is noise
        if self.use_cuda:
            noise = Variable(torch.randn(deconstructed.shape).cuda())
        else:
            noise = Variable(torch.randn(deconstructed.shape))

        #initialize the network with the chosen architecture
        image_channels = deconstructed.shape[1]
        net = PixelShuffleHourglass(image_channels)

        #bind the network to the gpu if cuda is enabled
        if self.use_cuda:
            net.cuda()

        #network optimizer set up
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

        result = None
        for step in range(self.num_steps):
            #get the network output
            output = net(noise)
            #we are only concerned with the output where we have the image available.
            masked_output = output * mask
            # calculate the l2_loss over the masked output and take an optimizer step
            optimizer.zero_grad()
            loss = torch.sum((masked_output - deconstructed) ** 2)
            loss.backward()
            optimizer.step()

            loss_value = loss.data.cpu().tolist()
            print('At step {}, loss is {}'.format(step, loss_value))

            result = output.data

            if loss_value < self.min_loss:
                return result

            if self.use_cuda:
                noise.data += self.sigma * torch.randn(noise.shape).cuda()
            else:
                noise.data += self.sigma * torch.randn(noise.shape)

        #clean up any mess we're leaving on the gpu
        if self.use_cuda:
            torch.cuda.empty_cache()

        return result

#define an encoder decoder network with pixel shuffle upsampling
class PixelShuffleHourglass(nn.Module):
    def __init__(self, image_channels):
        super(PixelShuffleHourglass, self).__init__()
        self.d_conv_1 = nn.Conv2d(image_channels, 8, 5, stride=2, padding=2)
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

        self.out_conv = nn.Conv2d(4, image_channels, 5, stride=1, padding=2)
        self.out_bn = nn.BatchNorm2d(image_channels)

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

        up_5 = F.pixel_shuffle(down_6, 2)
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
