# dataset for test set without labels

import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import csv
import cv2
import numpy as np
import torch

class CustomImageDataset(Dataset):  #inherent from dataset class

    def __init__(self, label_title_dir, img_dir, transform_1=None, transform_2=None):
        self.label_title_dir = pd.read_csv(label_title_dir)
        self.img_dir = img_dir
        self.transform_1 = transform_1
        self.transform_2 = transform_2


    def __len__(self):
        return len(self.label_title_dir)

    def __getitem__(self, index):

        img_path = os.path.join(self.img_dir, self.label_title_dir.iloc[index, 0])
        image = cv2.imread(img_path)

        image_bgr = cv2.split(image)
        image_nparray = np.array(image_bgr)
        image_tensor = torch.FloatTensor(image_nparray)


        if self.transform_1:
            image_tensor = self.transform_1.transform_1()(image_tensor)

            image_nparray = np.array(image_tensor)

        if self.transform_2:
            image_nparray = self.transform_2.transform_2(image_nparray)


        image_nparray = np.transpose(image_nparray, (2, 0, 1))
        image_tensor = torch.FloatTensor(image_nparray)

        return image_tensor # all tensors