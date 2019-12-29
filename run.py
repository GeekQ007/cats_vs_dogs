# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 22:50:01 2019

@author: Zero
"""

import torch
from torch.autograd import Variable
from torchvision import datasets, transforms, utils
import os
from tensorboardX import SummaryWriter
from lenet import LeNet5


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

# 图片导入
path = './dogs_vs_cats/'
transform = transforms.Compose([transforms.CenterCrop(225),
                                transforms.ToTensor()])
data_image = {x: datasets.ImageFolder(root=os.path.join(path, x),
                transform=transform)
                for x in ["train", "valid"]}
data_loader_image = {x: torch.utils.data.DataLoader(dataset=data_image[x],
                        batch_size=64,
                        shuffle=True)
                        for x in ["train", "valid"]}
classes = data_image["train"].classes
images, labels = next(iter(data_loader_image["train"]))
# 打印image_train图像中对应的label_train，也就是图像的类型
print([classes[i] for i in labels])
grid = utils.make_grid(images)

tb = SummaryWriter()
tb.add_image('images', grid)
tb.add_graph(LeNet5(), images)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = LeNet5().to(device)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)

for epoch in range (20):
    
    total_loss = 0
    total_correct = 0

    for batch, data in enumerate(data_loader_image["train"], 0):
        images, labels = data
        images, labels = Variable(images).to(device), Variable(labels).to(device)
 
        optimizer.zero_grad()

        preds = network(images)
        
        batch_loss = loss(preds, labels)
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
        total_correct += get_num_correct(preds, labels)
        if(batch % 5 == 0):
            print('batch {},Loss {:.4f},Accuracy {:.4f}'
                  .format(batch, total_loss/((batch+1)*64), total_correct/((batch+1)*64)))
    print('epoch {},Loss {:.4f},Accuracy {:.4f}'
          .format(epoch, total_loss/len(data_image["train"]), total_correct/len(data_image["train"])))
    tb.add_scalar('Loss', total_loss/len(data_image["train"]), epoch)
    tb.add_scalar('Accuracy', total_correct/len(data_image["train"]), epoch) 
torch.save(network, "./lenet5_catvsdog.pth")
tb.close()
        
        
        
        

