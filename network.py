# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 00:18:22 2019

@author: Zero
"""

import torch.nn as nn
import torchvision.models as models


class feature_net(nn.Module):
    def __init__(self, model, dim, n_classes):
        super(feature_net, self).__init__()
        if model == 'vgg':
            vgg = models.vgg19(pretrained=True)
            self.feature = nn.Sequential(*list(vgg.children())[:-1])
            self.feature.add_module('global average', nn.AvgPool2d(7))
        elif model == 'inceptionv3':
            inception = models.inception_v3(pretrained=True)
            self.feature = nn.Sequential(*list(inception.children())[:-1])
            self.feature._modules.pop('13')
            self.feature.add_module('global average', nn.AvgPool2d(7))
        elif model == 'resnet152':
            resnet = models.resnet152(pretrained=True)
            self.feature = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
