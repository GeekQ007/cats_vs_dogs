import torch.nn as nn
from collections import OrderedDict


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3, 64, kernel_size=(3, 3), stride=2, padding=0)), #112*112*64
            ('ReLu1', nn.ReLU()),
            ('p2', nn.MaxPool2d(kernel_size=(2, 2))), #56*56*64
            ('c3', nn.Conv2d(64, 128, kernel_size=(5, 5), stride=3, padding=0)), #18*18*128
            ('ReLu2', nn.ReLU()),
            ('p4', nn.MaxPool2d(kernel_size=(2, 2))), #9*9*128
            ('c5', nn.Conv2d(128, 256, kernel_size=(9, 9), padding=0)), #4*4*256 
            ('ReLu3', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(256, 128)),
            ('ReLu4', nn.ReLU()),
            ('f7', nn.Linear(128, 2)),
            ('sig', nn.Softmax(dim=1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output
