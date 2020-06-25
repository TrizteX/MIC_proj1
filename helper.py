from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

imsize = 512 if (torch.cuda.is_available()) else 128  # use small size if no gpu

cnn = models.vgg19(pretrained=True).features.to(device).eval()

"""VGG networks are trained on images with each channel normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
"""

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
            


def image_loader(cnt,sty):
    styimg = Image.open(sty)
    cntimg= Image.open(cnt)

    w,h=cntimg.size
    ratio=w/h

    l=max(w,h)
    s=min(w,h)

    if l>imsize:
        s=int(s/l*imsize)
        l=imsize
    if ratio==(s/l): 
        styimg=styimg.resize((s,l))
        cntimg=cntimg.resize((s,l))
    elif ratio==(l/s):
        styimg=styimg.resize((l,s))
        cntimg=cntimg.resize((l,s))
        
    loader = transforms.Compose([transforms.ToTensor()])

    # fake batch dimension required to fit network's input dimensions
    content_image = loader(cntimg).unsqueeze(0)
    style_image = loader(styimg).unsqueeze(0)
    input_img = torch.randn(content_image.data.size(), device=device)
    
    return content_image.to(device, torch.float),style_image.to(device, torch.float),input_img

#displays an image by reconverting a tensor to an image
def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    plt.ion()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated
    
 class ContentLoss(nn.Module):

        def __init__(self, target,):
            super(ContentLoss, self).__init__()
              # we 'detach' the target content from the tree used
              # to dynamically compute the gradient: this is a stated value,
              # not a variable. Otherwise the forward method of the criterion
              # will throw an error.
            self.target = target.detach()

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input


"""Style Loss"""
      #GRAM MATRIX

def gram_matrix(input):
        a, b, c, d = input.size()  # a=batch size(=1)
          # b=number of feature maps
          # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

          # we 'normalize' the values of the gram matrix
          # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

class StyleLoss(nn.Module):

        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input

"""Importing the Model
      -------------------
      Import a pre-trained neural network. We will use a 19 layer VGG network like the one used in the paper.

"""

    

      # create a module to normalize input image so we can easily put it in a
      # nn.Sequential
class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
              # .view the mean and std to make them [C x 1 x 1] so that they can
              # directly work with image Tensor of shape [B x C x H x W].
              # B is batch size. C is number of channels. H is height and W is width.
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
              # normalize img
            return (img - self.mean) / self.std
