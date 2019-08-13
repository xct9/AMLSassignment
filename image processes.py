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

# face detector.
predictor_path = 'dlib/shape_predictor_5_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()

# convert into ImageFolderDataset for pytorch dataloader.
# https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.ImageFolder
def convert(X, y, phase, attr):
    for file_id, class_index in zip(X, y):
        file_name = '{}.png'.format(file_id)
        file_path = os.path.join('face_data/dataset', file_name)
        class_path = os.path.join('output', attr, phase, str(class_index))
        if not os.path.exists(class_path):
            os.makedirs(class_path, exist_ok=True)
        # remove noisy images by face detection.
        img = dlib.load_rgb_image(file_path)
        dets = detector(img, 1)
        # sometimes no faces detected.
        if len(dets) != 1:
            continue
        shutil.copyfile(file_path, osp.join(class_path, file_name))
        
print('Starting training, it will comsume much of time. On CPU, 1 epoch takes about 1 minute...')
# use tensorboard to record training process including testing and training accuracy and loss.
# saved into log dir.
writer = SummaryWriter('log')
# best weights for every attr.
best_weights = dict()
# start training every attr.
for attr in df.columns:
    # split dataset into testing and training dataset with 1/5 of whole dataset to be testing and rest to be training.
    # random_state ensures generating same result every time.
    X_train, X_test, y_train, y_test = train_test_split(df.index.values,
                                                        df[attr], test_size=test_size, random_state=0)
    print('Start process images...')
    # conver into ImageFolder, and saved into output/{attr}/{training or testing}/{class_name}
    convert(X_train, y_train, 'train', attr)
    convert(X_test, y_test, 'test', attr)
    # using pretrained alexnet, and replace last classifier of the number of classes outputs.
    model = models.alexnet(pretrained=True)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, df[attr].unique().shape[0])
    model = model.to(device)
    # SGD optimizer and Cross Entropy Loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss()
    # training and testing dataset from folder generated from above step.
    test_dataset = torchvision.datasets.ImageFolder(osp.join('output', attr, 'test'), transform=test_transform)
    train_dataset = torchvision.datasets.ImageFolder(osp.join('output', attr, 'train'), transform=train_transform)

    # pytorch load data.
    dataloaders = dict()
    dataloaders['test'] = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size, num_workers=8)
    dataloaders['train'] = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    best_test_acc = 0.
    best_weights[attr] = copy.deepcopy(model)
    # start training for attr.
    for epoch in range(epochs):
        print('{}, Epoch {}/{}'.format(attr, epoch+1, epochs))
        # including test and train.
        for phase in ['train', 'test']:
            total_loss, nums, correct = 0.0, 0, 0
            st_time = time.time()
            # for every batch.
            for images, labels in dataloaders[phase]:
                images, labels = images.to(device), labels.to(device)
                # only training will backward and update weight in neural networ.
                if phase == 'train':
                    model.train()
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                # testing will predicts the testing data and compute accuracy.
                else:
                    model.eval()
                    with torch.no_grad():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                nums += images.shape[0]
                total_loss += loss.item() * images.shape[0]
                prob, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
            # save the best result.
            acc = correct/nums
            if phase == 'test' and acc > best_test_acc:
                best_test_acc = acc
                best_weights[attr] = model
            print('{}: loss {:.5f}, takes {:.2f}s, acc {:.3f}'.format(
                phase, total_loss / nums, time.time() - st_time, acc
            ))

            # write into tensorboard.
            writer.add_scalars('{}/loss'.format(attr), {phase: total_loss / nums}, epoch)
            writer.add_scalars('{}/accuracy'.format(attr), {phase: acc}, epoch)
        scheduler.step()

writer.close()

print('training ended...')
print('in project folder, runs: tensorboard --logdir=log, it wll open the tensorboard panel...')
base_path = 'face_data/testing_dataset'

# output prediction for every attr and every test image.
output_csv_file = []
for file_name in os.listdir(base_path):
    single_output = dict()
    single_output['file_name'] = int(file_name.rstrip('.png'))
    # process testing data
    image = test_transform(Image.open(os.path.join(base_path, file_name))).unsqueeze(0).to(device)
    with torch.no_grad():
        # output for every attr.
        for attr in best_weights.keys():
            best_weights[attr].eval()
            outputs = best_weights[attr](image)
            _, preds = torch.max(outputs, 1)
            class_names = np.array(sorted(os.listdir(osp.join('output', attr, 'train'))))
            single_output[attr] = class_names[preds.cpu().numpy()][0]

    output_csv_file.append(single_output)

# convert into pandas dataframe and save into csv file.
df = pd.DataFrame.from_dict(output_csv_file)
df = df.set_index('file_name').sort_index()

output_csv_path = 'face_data/testing_dataset_output.csv'
df.to_csv(output_csv_path)
print('testing data outputs to {}'.format(output_csv_path))
