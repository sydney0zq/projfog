import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
import matplotlib .pyplot as plt
import os
import time
from PIL import Image, ImageFile
import torch.nn.functional as F

# Load data
data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
}

data_dir = "./"
im_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(im_datasets[x], batch_size=8, shuffle=True, num_workers=4)
                for x in ['train', 'valid']}

dataset_sizes = {x: len(im_datasets[x]) for x in ['train', 'valid']}

ImageFile.LOAD_TRUNCATED_IMAGES = True

use_gpu = torch.cuda.is_available()
print (" | CUDA available is {}".format(use_gpu))

def train_model(model, criterion, optimizer, scheduler, n_epochs=25):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_loss = float('inf')

    for epoch in range(n_epochs):
        print (" | Epoch {}/{}".format(epoch, n_epochs-1))
        print (" | " + "-" * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)   # Set model in training mode
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]: # Iter over data
                # Get the inputs
                inputs, labels = data
                # Wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Foward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                #print (" | Loss: ", loss.data[0])

                # Backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # Statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
           
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print (' | {} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Deep copy of the model
            if phase == 'valid' and epoch_acc >= best_acc and best_loss >= epoch_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, "./resnet152_avg.pth.tar")
                print (" | Epoch {} state saved, now acc reaches {}...".format(epoch, best_acc))
        print (" | Time consuming: {:.2f}s".format(time.time()-since))
        print (" | ")

model = models.resnet152(pretrained=True)

class novelmodel(nn.Module):
    def __init__(self):
        super(novelmodel, self).__init__()
        self.features = nn.Sequential(
            *list(model.children())[:-2]
        )
        self.conv1 = torch.nn.Conv2d(2048, 2, kernel_size=(1, 1), stride=2)
        self.avgpool = torch.nn.AvgPool2d(4)
    def forward(self, x):
        #print ("Feature size: {}".format(x.size()))
        x = self.features(x)
        x = self.conv1(x)
        #print ("Conv1 size: {}".format(x.size()))
        x = self.avgpool(x)
        #x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = x[:, :, 0, 0]
        return x

model = novelmodel()

if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, n_epochs=50)


