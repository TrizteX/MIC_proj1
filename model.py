import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.utils
import numpy as np
import argparse
import os
import subprocess as sp
from wgan import *
import json
import gdown
from train import *
class GAN:
    def __init__(self, run_type = "train"):
        print("run_type = ",run_type)
        if run_type == "inference":
            set_ckpt_dir = "WGAN-gen.pt"  # path to ckpt
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            def gen():
                set_gen_dir = "gen"  # path to save img directory
                if os.path.exists(set_gen_dir):
                    print("Found gen directory")
                    return 1
                else:
                    print("Directory for saving images not found, making one")
                    os.mkdir("gen")
                    set_gen_dir = "gen"
                    return 1

            def check_weights():
                if os.path.exists(set_ckpt_dir):
                    print("Found weights")
                    return 1
                else:
                    print("Downloading weigths")
                    download_weights()

            def download_weights():
                with open("config/weights_download.json") as fp:
                    json_file = json.load(fp)
                    url = 'https://drive.google.com/uc?id={}'.format(json_file['WGAN-gen.pt'])
                    gdown.download(url, "WGAN-gen.pt", quiet=False)
                    set_ckpt_dir = "WGAN-gen.pt"
                    print("Download finished")

            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            check_weights()
            gen()
            gan = WGAN()
            gan.eval()
            gan = gan.to(device)
            gan.load_model(filename=set_ckpt_dir)

            def save_new_img():
                len = 20  # number of images to be generated
                for i in range(len):
                    vec = gan.create_latent_var(1, random.randint(1, 200))  # batch, seed value
                    img = gan.generate_img(vec)
                    img = unnormalize(img)
                    fname_in = '{}/frame{}.png'.format("gen", i)
                    torchvision.utils.save_image(img, fname_in, padding=0)
                print("All images are saved in gen")

            save_new_img()


        elif run_type == "train":
            model = CelebA(train_params, ckpt_params, gan_params)
            data_loader = wgan.load_dataset()

            torch.manual_seed(100)
            n_epoch = 135  # Number of epochs to train for
            model.train(n_epoch, data_loader)

if __name__ == "__main__":
    GAN()
