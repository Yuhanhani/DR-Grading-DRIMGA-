# dataset used for multitasking

import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import csv
import cv2
import numpy as np
import torch
from torchvision import transforms

class CustomImageDataset(Dataset):  # inherent from dataset class

    def __init__(self, annotations_file, img_dir, transform_1=None, transform_2=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform_1 = transform_1
        self.transform_2 = transform_2
        self.target_transform = target_transform


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index, 0])
        image = cv2.imread(img_path)

        image_bgr = cv2.split(image)  # split the 1D BGR image into three channels
        image_nparray = np.array(image_bgr)
        image_tensor = torch.FloatTensor(image_nparray)  # [3, 1024, 1024]


        label_task2 = self.img_labels.iloc[index, 1]
        label_task3 = self.img_labels.iloc[index, 2]
        label = [label_task2, label_task3]


        if self.transform_1:
            image_tensor = self.transform_1.transform_1()(image_tensor)  # PIL image for with toPIL, tensor for without case

        image_nparray = np.array(image_tensor)  # array, [1024, 1024, 3] for with toPIL, [3, 1024, 1024] for without case

        # image_nparray = np.transpose(image_nparray, (1, 2, 0))  # [ 1024, 1024, 3] only use for without case

        if self.transform_2:
            image_nparray = self.transform_2.transform_2(image_nparray)
        if self.target_transform:
            label = self.target_transform(label)


        image_nparray = np.transpose(image_nparray, (2, 0, 1))
        image_tensor = torch.FloatTensor(image_nparray)  # [3, 1024, 1024]
                                                         # convert image to tensor (already tensor, but just to ensure)


        # print(index)

        y_true = torch.tensor([label])  # convert label to tensor


        return image_tensor, y_true   # all tensors, y_true now is a tensor with each element a two-element list


    # remember to permute back