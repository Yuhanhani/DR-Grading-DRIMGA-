# multi-tasking of task 2 and 3
# split dataset according to overall label distributions
# also can implement focal loss
# incorporate cross validation ("test" here is equivalent to "validate")
# also incorporating provided metrics
# with data augmentation
# with test loop
# with saving model, adaptive learning rate (learning rate scheduler) & dataset class & saving model

import os
from PIL import Image
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
from torchvision import transforms
import torchvision.models as models
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
from torch import nn
import pandas as pd
import csv
import cv2
import metrics
import metric_classification
from sklearn.model_selection import train_test_split
import random
import sklearn
import MT_dataset
import data_augmentation


device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(device)


# download and read labels and data ------------------------------------------------------------------------------------

pd.options.display.max_rows = 1000

if torch.cuda.is_available():
    annotations_file = '/well/papiez/users/pea322/MT/multi_label.csv'
else:
    annotations_file = '/Users/mirandazheng/Desktop/folder/labels.csv'

if torch.cuda.is_available():
    img_dir = '/well/papiez/users/pea322/MT/Original_Images'
else:
    img_dir = '/Users/mirandazheng/Desktop/folder'


transform_1 = data_augmentation.data_augmentation_transform(phase='train')
train_dataset = MT_dataset.CustomImageDataset(annotations_file=annotations_file, img_dir=img_dir, transform_1=transform_1)


# type: dataset.CustomImageDataset
length = train_dataset.__len__()


class0_data = []
index0 = []

class1_data = []
index1 = []

class2_data = []
index2 = []


train_data_pre = []
train_label_pre = []
index_quality = []
for i in range(length):

    image, label = train_dataset.__getitem__(i)

    train_data_pre.append([image, label])

    if label[0, 1] == 0:  # same as [0][1]
        index0.append(i)  # 303
    elif label[0, 1] == 1:
        index1.append(i)  # 195
    elif label[0, 1] == 2:
        index2.append(i)  # 64
    elif label[0, 1] == 3:
        index_quality.append(i)  # 50


random_state = 7

train_index0, test_index0 = train_test_split(index0, test_size=0.085, random_state=random_state)  # 277, 26
train_index1, test_index1 = train_test_split(index1, test_size=0.087, random_state=random_state)  # 178, 17
train_index2, test_index2 = train_test_split(index2, test_size=0.093, random_state=random_state)  # 58, 6
train_index_quality, test_index_quality = train_test_split(index_quality, test_size=0.08, random_state=random_state)  # 46, 4


train_index = train_index0 + train_index1 + train_index2 + train_index_quality # 559
test_index = test_index0 + test_index1 + test_index2 + test_index_quality  # 53

random.seed(random_state)    # or use shuffle = True in Dataloader
random.shuffle(train_index)  # changes original list, no new return value
random.shuffle(test_index)

train_data = []
train_label = []
for i in range(len(train_index)):

    image, label = train_data_pre[train_index[i]]

    train_data.append([image, label])

    if i == 0:
        train_label = label
    else:
        train_label = torch.cat((train_label, label), 0)

print(train_label)
print(train_label.size()) # torch.Size([559, 2])
print(type(train_label)) # type: torch.Tensor


transform_1 = data_augmentation.data_augmentation_transform(phase='test')
test_dataset = MT_dataset.CustomImageDataset(annotations_file=annotations_file, img_dir=img_dir, transform_1=transform_1)

test_data = []
test_label = []
for i in range(len(test_index)):

    image, label = test_dataset.__getitem__(test_index[i])

    test_data.append([image, label])

    if i == 0:
        test_label = label
    else:
        test_label = torch.cat((test_label, label), 0)

print(test_label)
print(test_label.size())  # torch.Size([53, 2])
print(type(test_label))  # type: torch.Tensor

# or write index into dataset class and load dataset to Dataloader

batch_size = 25
train_dataloader = DataLoader(train_data, batch_size=batch_size) # type: torch.utils.data.dataloader.DataLoader
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# use model-------------------------------------------------------------------------------------------------------------

model = models.resnet34(weights='ResNet34_Weights.DEFAULT', progress=False)
model.fc = nn.Linear(512 * 1, 6)


# ----------------------------------------------------------------------------------------------------------------------
# cannot recover the 20th metrics value because this model is saved after last back propagation, but the metrics are
# calculated before 20th back propagation. can save model also at this position or just keep it and test using
# training data
# path = '/well/papiez/users/pea322/seed1/resnet34_model/1_25_0.001_adaptive_0.9_9_augTHVRJS512_L10.21.pth'
# # path = '/users/papiez/pea322/resnet34_model/15_0.001_constant_20_pretrained_(2).pth'
# model.load_state_dict(torch.load(path))
# model.eval() batch normalisation makes training descent smoothier, without it, could use smaller learning rate to try
# to see the same effect. Otherwise the loss will explode
# ----------------------------------------------------------------------------------------------------------------------


model = model.to(device)  # move model to GPU

coe = 0.01 # or 0.01



# Define the training loop ---------------------------------------------------------------------------------------------

def train_loop(dataloader, model, loss_fn_task2, loss_fn_task3, optimizer):

    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):

        X = X.to(device)
        y = y.to(device)

        X_one_label = X[y[:, 0, 1] == 3] # torch.size([4,3, 256,256])
        y_one_label = y[y[:, 0, 1] == 3] # torch.size([4,1,2])

        X_two_label = X[y[:, 0, 1] != 3]# torch.size([21, 3, 256,256])
        y_two_label = y[y[:, 0, 1] != 3] # torch.size([21,1,2])

        pred_one_label = model(X_one_label) # torch.size([4,6])
        pred_two_label = model(X_two_label)

        pred_one_label = pred_one_label[:, 0:6]  # only for vit
        pred_two_label = pred_two_label[:, 0:6]

        # print(pred_one_label.size())

        loss_1 = loss_fn_task3(pred_two_label[:, 3:], y_two_label[:, 0, 1])  # grading loss
        loss_2 = loss_fn_task2(pred_two_label[:, 0:3], y_two_label[:, 0, 0]) + loss_fn_task2(pred_one_label[:, 0:3], y_one_label[:, 0, 0]) # quality loss

        loss = loss_1 + coe * loss_2

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch == int(size//batch_size):
            loss, current = loss.item(), batch * batch_size
        else:
            loss, current = loss.item(), batch * len(X)

        print(f'loss:{loss:>7f} [{current:>5d}/{size:>5d}]')



# Define test loop -----------------------------------------------------------------------------------------------------

def test_loop(dataloader, model, loss_fn_task2, loss_fn_task3):

    size = len(dataloader.dataset)
    total_loss = 0
    prediction = torch.empty(batch_size, 6)

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):

            X = X.to(device)
            y = y.to(device)

            # print(X.size())  # torch.size([25,3,256, 256])
            # print(y.size())  # torch.size([25,1,2])
            # print(y[3, 0, 1])

            pred = model(X) # torch.szie([25,6])

            pred = pred[:, 0:6]  # only for vit

            initial_pred = pred

            if batch == 0:
                prediction = initial_pred
            else:
                prediction = torch.cat((prediction, pred), 0)  # order will remain the same since no stochasticity here

            del X, y

    return prediction


# Start to train (an example) ------------------------------------------------------------------------------------------

learning_rate = 0.001 # set the initial learning rate
epochs = 20
gamma = 0.9

# weight = [70/611, 213/611, 328/611]
# weight_task2 = [518/665, 97/665, 50/665]  # Lrcd
weight_task3 = [1, 3, 2]
weight_task2 = [1, 1, 1]
# weight_task3 = [0.25, 0.45, 0.3]
weight_task2 = torch.FloatTensor(weight_task2)
weight_task3 = torch.FloatTensor(weight_task3)
weight_task2 = weight_task2.to(device)
weight_task3 = weight_task3.to(device)

gamma_loss = 2.5
alpha = [70/611, 212/611, 329/611]
gamma_loss = torch.tensor(gamma_loss)
alpha = torch.tensor(alpha)
gamma_loss = gamma_loss.to(device)
alpha = alpha.to(device)

# loss_fn = focal_loss.FocalLoss(gamma=gamma_loss, alpha=alpha, reduction='sum')

loss_fn_task2 = nn.CrossEntropyLoss(weight=weight_task2, reduction='sum')
loss_fn_task3 = nn.CrossEntropyLoss(weight=weight_task3, reduction='sum')


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


if torch.cuda.is_available():
    project_path = '/well/papiez/users/pea322/densenet121_running'
else:
    project_path = '/Users/mirandazheng/PycharmProjects/pythonProject3'

for t in range(epochs):
    print(f'Epoch {t+1}\n train loop---------------------------- ')

    model.train()
    train_loop(train_dataloader, model, loss_fn_task2, loss_fn_task3,  optimizer)

    path = os.path.join(project_path, '{}_{}_{}_{}_{}_{}_{}.pth'.format(random_state, batch_size, learning_rate, 'adaptive', gamma, (t+1), 'aug.'))

    # torch.save(model.state_dict(), path)

    print(f'Epoch {t+1}\n test loop---------------------------- ')

    model.eval()
    prediction = test_loop(test_dataloader, model, loss_fn_task2, loss_fn_task3)

    scheduler.step()
    print('epoch={}, learning rate={:.6f}'.format(t+1, optimizer.state_dict()['param_groups'][0]['lr']))

    # print(prediction.size()) # torch.szie([53,6])

    Softmax_layer = nn.Softmax(dim=1)
    prediction_task2 = Softmax_layer(prediction[:, 0:3])
    prediction_task2 = prediction_task2.to(device)
    # print(prediction_task2)
    y_pred_task2 = torch.argmax(prediction_task2, dim=1)
    # print(y_pred_task2)
    prediction_task2 = prediction_task2.detach().cpu()
    y_pred_task2 = y_pred_task2.detach().cpu() # only move into cpu, can then be transferred into np array

    Softmax_layer = nn.Softmax(dim=1)
    prediction_task3 = Softmax_layer(prediction[:, 3:])
    prediction_task3 = prediction_task3.to(device)
    # print(prediction_task3)
    y_pred_task3 = torch.argmax(prediction_task3, dim=1)
    # print(y_pred_task3)
    prediction_task3 = prediction_task3.detach().cpu()
    y_pred_task3 = y_pred_task3.detach().cpu() # only move into cpu, can then be transferred into np array


    # evaluation for task 2 ----------------------------------------------------

    test_label_task2 = test_label[:, 0] # 1D tensor

    print(y_pred_task2)
    print(test_label_task2)

    kappa_score_task2 = metric_classification.quadratic_weighted_kappa(test_label_task2, y_pred_task2)
    macro_auc_task2 = metric_classification.roc_auc_score(test_label_task2, prediction_task2, average="macro", multi_class='ovo')
    macro_precision_task2 = metrics.marco_precision(test_label_task2, y_pred_task2)
    macro_sensitivity_task2 = metrics.marco_sensitivity(test_label_task2, y_pred_task2)
    macro_specificity_task2 = metrics.marco_specificity(test_label_task2, y_pred_task2)

    print(kappa_score_task2)
    print(macro_auc_task2)
    print(macro_precision_task2)
    print(macro_sensitivity_task2)
    print(macro_specificity_task2)

    cf_matrix_task2 = sklearn.metrics.confusion_matrix(test_label_task2, y_pred_task2)
    print(cf_matrix_task2)

    # evaluation for task 3 ----------------------------------------------------

    test_label_task3_medium = test_label[:, 1] # 1D tensor
    test_label_task3 = test_label_task3_medium[test_label_task3_medium != 3]
    y_pred_task3 = y_pred_task3[test_label_task3_medium != 3]
    prediction_task3 = prediction_task3[test_label_task3_medium != 3]
    print(prediction_task3.size())
    # print(test_label_task3.size()) torch.Size([49])

    print(y_pred_task3)
    print(test_label_task3)

    kappa_score_task3 = metric_classification.quadratic_weighted_kappa(test_label_task3, y_pred_task3)
    macro_auc_task3 = metric_classification.roc_auc_score(test_label_task3, prediction_task3, average="macro", multi_class='ovo')
    macro_precision_task3 = metrics.marco_precision(test_label_task3, y_pred_task3)
    macro_sensitivity_task3 = metrics.marco_sensitivity(test_label_task3, y_pred_task3)
    macro_specificity_task3 = metrics.marco_specificity(test_label_task3, y_pred_task3)

    print(kappa_score_task3)
    print(macro_auc_task3)
    print(macro_precision_task3)
    print(macro_sensitivity_task3)
    print(macro_specificity_task3)

    cf_matrix_task3 = sklearn.metrics.confusion_matrix(test_label_task3, y_pred_task3)
    print(cf_matrix_task3)


print('Done!')