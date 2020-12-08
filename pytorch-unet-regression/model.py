import os
import numpy as np

import torch
import torch.nn as nn

## UNet
class UNet(nn.Module):
    def __init__(self, nch, nker, norm="bnorm"):
        super(UNet, self).__init__()

        # Convolution, Batch_normalization, ReLU
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=0.0):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            
            if not norm is None:
                if norm == "bnorm":
                    layers += [nn.BatchNorm2d(num_features=out_channels)]
                elif norm == "inorm":
                    layers += [nn.InstanceNorm2d(num_features=out_channels)]

            if not relu is None:
                layers += [nn.ReLU() if relu == 0.0 else nn.LeakyReLU(relu)]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path (encoder)
        self.enc1_1 = CBR2d(in_channels=nch, out_channels=1 * nker) # , kernel_size=3, stride=1, padding=1, bias=True
        self.enc1_2 = CBR2d(in_channels=1 * nker, out_channels=1 * nker)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=1 * nker, out_channels=2 * nker)
        self.enc2_2 = CBR2d(in_channels=2 * nker, out_channels=2 * nker)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=2 * nker, out_channels=4 * nker)
        self.enc3_2 = CBR2d(in_channels=4 * nker, out_channels=4 * nker)
        
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=4 * nker, out_channels=8 * nker)
        self.enc4_2 = CBR2d(in_channels=8 * nker, out_channels=8 * nker)
        
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=8 * nker, out_channels=16 * nker)
        
        # Expansive path (decoder)
        self.dec5_1 = CBR2d(in_channels=16 * nker, out_channels=8 * nker)
        
        self.unpool4 = nn.ConvTranspose2d(in_channels=8 * nker, out_channels=8 * nker, 
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 8 * nker, out_channels=8 * nker)
        self.dec4_1 = CBR2d(in_channels=8 * nker, out_channels=4 * nker)

        self.unpool3 = nn.ConvTranspose2d(in_channels=4 * nker, out_channels=4 * nker, 
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 4 * nker, out_channels=4 * nker)
        self.dec3_1 = CBR2d(in_channels=4 * nker, out_channels=2 * nker)

        self.unpool2 = nn.ConvTranspose2d(in_channels=2 * nker, out_channels=2 * nker, 
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 2 * nker, out_channels=2 * nker)
        self.dec2_1 = CBR2d(in_channels=2 * nker, out_channels=1 * nker)

        self.unpool1 = nn.ConvTranspose2d(in_channels=1 * nker, out_channels=1 * nker,
                                        kernel_size=2, stride=2, padding=0, bias=True)
                                        
        self.dec1_2 = CBR2d(in_channels=2 * 1 * nker, out_channels= 1 * nker)
        self.dec1_1 = CBR2d(in_channels=1 * nker, out_channels=1 * nker)
        
        # output map channel을 2개 + nn.CrossEntropyLoss == output map channel을 1개 + nn.BCELoss(binary cross entropy loss)
        # self.fc = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc = nn.Conv2d(in_channels=1 * nker, out_channels=nch, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        

        dec5_1 = self.dec5_1(enc5_1)
        
        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        # concatenation, dim=[0: batch, 1: channel, 2: height, 3: width]
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x
