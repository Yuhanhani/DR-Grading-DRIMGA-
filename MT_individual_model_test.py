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
# from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import data_augmentation
import metric_classification
from sklearn.model_selection import train_test_split
import random
import sklearn
import MT_dataset
import data_augmentation
import focal_loss

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)


# download and read labels and data ------------------------------------------------------------------------------------

pd.options.display.max_rows = 1000

if torch.cuda.is_available():
    annotations_file = '/well/papiez/users/pea322/MT_self_test/self_test_multi_label.csv'
else:
    annotations_file = '/Users/mirandazheng/Desktop/folder/labels.csv'

if torch.cuda.is_available():
    img_dir = '/well/papiez/users/pea322/MT_self_test/Original_Images'
else:
    img_dir = '/Users/mirandazheng/Desktop/folder'


transform_1 = data_augmentation2.data_augmentation_transform(phase='test')
test_dataset = MT_dataset.CustomImageDataset(annotations_file=annotations_file, img_dir=img_dir, transform_1=transform_1)

test_data = []
test_label = []
for i in range(test_dataset.__len__()):

    image, label = test_dataset.__getitem__(i)

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
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# use model-------------------------------------------------------------------------------------------------------------
# model = models.resnet152(weights='ResNet152_Weights.DEFAULT', progress=False) # equivalent to parameter = false
# model.fc = nn.Linear(512 * 4, 6)   # can use this to change fully connected -> 1000 to 3
#                                    # dimension of last layer (input and output)

# model = models.resnet34(weights=True, progress=False) # equivalent to parameter = false
model = models.resnet34(weights='ResNet34_Weights.DEFAULT', progress=False)
model.fc = nn.Linear(512 * 1, 6)

# model = models.resnext50_32x4d(weights=None, progress=True)
# model.load_state_dict(torch.load('/well/papiez/users/pea322/weights/resnext50_32x4d-7cdf4587.pth'))
# model.fc = nn.Linear(512 * 4, 6)

# model = models.wide_resnet50_2(weights=None, progress=True)
# model.load_state_dict(torch.load('/well/papiez/users/pea322/weights/wide_resnet50_2-95faca4d.pth'))
# model.fc = nn.Linear(512 * 4, 6)

# model = models.densenet121(weights='DenseNet121_Weights.DEFAULT', progress=True)
# model.classifier = nn.Linear(1024, 6) #1024, 2208, 1664, 1920

# model = models.inception_v3(weights='Inception_V3_Weights.DEFAULT', progress=False)
# model.AuxLogits.fc = nn.Linear(768, 3)
# model.fc = nn.Linear(2048, 3)

# model = models.squeezenet1_1(weights='SqueezeNet1_1_Weights.DEFAULT', progress=False)
# model.classifier[1] = nn.Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1))

# model = models.vgg16_bn(weights=None, progress=False) # if no bn, gives infinity loss
# model.load_state_dict(torch.load('/well/papiez/users/pea322/weights/vgg16_bn-6c64b313.pth'))
# model.classifier[6] = nn.Linear(4096, 6)

# model = models.vgg19_bn(weights='VGG19_BN_Weights.DEFAULT', progress=False) # if no bn, gives infinity loss
# model.classifier[6] = nn.Linear(4096, 6)

# model = models.vgg11_bn(weights=None, progress=False) # if no bn, gives infinity loss
# model.load_state_dict(torch.load('/well/papiez/users/pea322/weights/vgg11_bn-6002323d.pth'))
# model.classifier[6] = nn.Linear(4096, 6)

# model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT', progress=False)
# model.classifier[1] = nn.Linear(in_features=1280, out_features=6)  # 1280, 2560

# model = models.efficientnet_v2_l(weights=None, progress=False)
# model.load_state_dict(torch.load('/well/papiez/users/pea322/weights/efficientnet_v2_l-59c71312.pth'))
# model.classifier[1] = nn.Linear(in_features=1280, out_features=6)

# ----------------------------------------------------------------------------------------------------------------------
# cannot recover the 20th metrics value because this model is saved after last back propagation, but the metrics are
# calculated before 20th back propagation. can save model also at this position or just keep it and test using
# training data
path = '/well/papiez/users/pea322/MT_seed7_new/resnet34_model/7_25_0.001_adaptive_0.9_5_aug.pth'
# # path = '/users/papiez/pea322/resnet34_model/15_0.001_constant_20_pretrained_(2).pth'
model.load_state_dict(torch.load(path))
# model.eval() batch normalisation makes training descent smoothier, without it, could use smaller learning rate to try
# to see the same effect. Otherwise the loss will explode
# ----------------------------------------------------------------------------------------------------------------------


model = model.to(device)  # move model to GPU


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

            initial_pred = pred

            if batch == 0:
                prediction = initial_pred
            else:
                prediction = torch.cat((prediction, pred), 0)  # order will remain the same since no stochasticity here

            del X, y

    return prediction


# Start to train (an example) ------------------------------------------------------------------------------------------

learning_rate = 0.001 # set the initial learning rate
epochs = 1
gamma = 0.8

# weight = [70/611, 213/611, 328/611]
weight_task2 = [1, 1, 1]
weight_task3 = [1, 3.5, 2]
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


for t in range(epochs):

    print(f'Epoch {t+1}\n test loop---------------------------- ')

    model.eval()
    prediction = test_loop(test_dataloader, model, loss_fn_task2, loss_fn_task3)

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