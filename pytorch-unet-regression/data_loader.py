import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from util import *

## Data Loader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, task=None, opts=None):
        self.data_dir = data_dir
        self.transform = transform
        self.task = task
        self.opts = opts

        lst_data = os.listdir(self.data_dir)

        lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('png')]

        lst_data.sort()

        self.lst_data = lst_data

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):
        # label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        # input = np.load(os.path.join(self.data_dir, self.lst_input[index]))
        img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
        size = img.shape

        if size[0] > size[1]: # 항상 가로로 긴 이미지로 저장
            img = img.transpose((1, 0, 2))

        if img.dtype == np.unit8:
            img = img/255.0

        if img.ndim == 2:
            label = label[:, :, np.newaxis]

        label = img

        if self.task == "denoising":
            input = add_noise(img, type=self.opts[0], opts=self.opts[1])
        elif self.task == "inpainting":
            input = add_sampling(img, type=self.opts[0], opts=self.opts[1])
        elif self.task == "super_resolution":
            input = add_blur(img, type=self.opts[0], opts=self.opts[1])


        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data

## Data Transform
# ToTensor(): numpy -> tensor
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # Image의 numpy 차원 = (Y, X, CH)
        # Image의 tensor 차원 = (CH, Y, X)
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']
        
        # regression은 label에도 적용해줘야함
        label = (label - self.mean) / self.std
        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data


class RandomCrop(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        input, label = data['input'], data['label']

        h, w = input.shape[:2]
        new_h, new_w = self.shape

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
        id_x = np.arange(left, left + new_w, 1)

        input = input[id+_y, id_x]
        label = label[id_y, id_x]

        data = {'input': input, 'label': label}

        return data