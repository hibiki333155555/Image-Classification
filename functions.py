# import ML libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
import glob
from PIL import Image
import tqdm

# import pytorch libraries for image classification
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models

# set config class
class Cfg():
    def __init__(self):
        self.batch_size = 32
        self.num_workers = 2
        self.num_epochs = 10
        self.lr = 0.001
        self.momentum = 0.9
        self.seed = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    
# class to load input image
# 画像の前処理をを行うクラス
# 訓練時と検証時で異なる動作をする (データオーギュメンテーション)
class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)
    

# make path list for train and validation
# 学習用データと検証用データのファイルパスを格納したリストを作成する
# train_list = make_datapath_list(phase='train')
# val_list = make_datapath_list(phase='val')
def make_datapath_list(phase='train'):

    rootpath = './'
    target_path = os.path.join(rootpath+phase+'/**/*.jpg')

    path_list = []
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


# make dataset class for train and validation
class UDataset(Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase)

        if self.phase == "train":
            label = img_path[:]
        elif self.phase == "val":
            label = img_path[:]

        if label == "":
            label = 0
        elif label == "":
            label = 1
        elif label == "":
            label = 2
        elif label == "":
            label = 3
        elif label == "":
            label = 4


        return img_transformed, label
    

# make Dataloader for train and validation
def make_dataloader(phase='train'):
    # make path list
    path_list = make_datapath_list(phase)

    # make dataset
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # make dataloader
    batch_size = 32
    
    dataset = Dataset(file_list=path_list, transform=ImageTransform(224, mean, std), phase=phase) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    dataloaders_dict = {"train": dataloader, "val": dataloader}

    return dataloaders_dict


# function to load VGG16 model
def load_vgg16_model():
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier[6] = nn.Linear(4096, 5)
    # 転移学習の場合、最後の層の重みだけ更新するので勾配計算は最後の層だけ可能な状態にしておく

    return model

# function to load ResNet50 model
def load_resnet50_model():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(2048, 5)

    return model


# train model
def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs):
    # initialize
    cfg = Cfg()
    device = cfg.device
    print("device:", device)
    torch.backends.cudnn.benchmark = True

    model.to(device)

    # train and test
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and (phase == 'train'):
                continue

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    # save model
    save_path = './weights/weights.pth'
    torch.save(model.state_dict(), save_path)


    return model

