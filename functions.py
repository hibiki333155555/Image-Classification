#import ML libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import glob
from PIL import Image

#import pytorch libraries for image classification
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models



# function to load VGG16 model
def load_vgg16_model():
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(4096, 5)

    return model

    
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

    return dataloader

