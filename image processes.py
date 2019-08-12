import torch
import torchvision
from torchvision import transforms, models
import pandas as pd
import os.path as osp
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
import os
import shutil
import numpy as np
import time
import copy
from PIL import Image
try:
    import dlib
except:
    print('dlib not found, plz install from https://github.com/davisking/dlib.')
    exit(0)

# training 20 epochs with 120 batchsize.
batch_size = 120
epochs = 20
# initial learning rate, decay with 0.7 gamma every 3 step size.
lr = 0.002
gamma = 0.7
step_size = 3
# training datat 0.8, testing data 0.2
test_size = 0.2
# use gpu if has cuda or GPU, default gpu device is cuda:0 or the first . 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# read from csv file.
# index_col=0 means using 'file_name' as index column.
df = pd.read_csv('face_data/attribute_list_hair_color_v2.csv', index_col=0)
if osp.exists('output'):
    shutil.rmtree('output', ignore_errors=True)
os.mkdir('output')

# data augment for training data, testing data normalized only.
train_transform = transforms.Compose([
    # random resize and crop images into 224*224, because pretrained alexnet needs the same input size.
    transforms.RandomResizedCrop(224),
    # random filp.
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # normalize image data by pretrained model needs.
    # reference https://pytorch.org/docs/stable/torchvision/models.html#classification
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    # testing images are croped from center area into 224*224.
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
