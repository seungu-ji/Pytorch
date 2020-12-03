import os
import numpy as np

import torch
import torch.nn as nn

## Network save
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
                "./%s/model_epoch%d.pth" % (ckpt_dir, epoch))

## Network load
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch
    
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('/%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

## Add Sampling
def add_sampling(img, type="random", opts=None):
    size = img.shape

    if type == "uniform":
        ds_y = opts[0].astype(np.int)
        ds_x = opts[1].astype(np.int)

        mask = np.zeros(size)
        mask[::ds_y, ::ds_x, :] = 1
        dst = img * mask
    elif type == "random":
        prob = opts[0]

        rnd = np.random.rand(size[0], size[1], size[2])
        mask = (rnd < prob).astype(np.float)
        dst = img * mask
    elif type == "gaussian":
        ly = np.linspace(-1, 1, size[0])
        lx = np.linspace(-1, 1, size[1])

        x, y = np.meshgrid(lx, ly)

        x0 = opts[0]
        y0 = opts[1]
        sgmx = opts[2]
        sgmy = opts[3]

        a = opts[4]

        gaus = a * np.exp(-((x - x0)**2/(2*sgmx**2) + (y - y0)**2/(2*sgmy**2)))
        gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, size[2]))
        rnd = np.random.rand(size[0], size[1], size[2])
        mask = (rnd < gaus).astype(np.float)
        
        dst = img * mask

    return dst

## Add Noise
def add_noise(img, type="random", opts=None):
    sz = img.shape

    if type == "random":
        sgm = opts[0]

        noise = sgm / 255.0 * np.random.randn(sz[0], sz[1], sz[2])

        dst = img + noise

    elif type == "poisson":
        dst = poisson.rvs(255.0 * img) / 255.0
        noise = dst - img

    return dst