from torchvision import transforms
import elasticdeform
import numpy
import torch
import warpim

class data_augmentation_transform():

    def __init__(self, phase):
        self.phase = phase

    def transform_1(self):

         transform_dict1 = {

            'train': transforms.Compose([
                transforms.ToPILImage(),
                # transforms.ToTensor(),cc
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=(0.8, 1), contrast=(1, 1.5), saturation=(1, 1), hue=(0, 0)),   #(1,1)x3, (0,0) return original images. small change has big influence on this image
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # actually belongs to transform not aug.
                                                                                      # use mean and std. calculated based on imagenet
                #transforms.RandomAdjustSharpness(sharpness_factor=5),
                #transforms.RandomInvert(),
                #transforms.RandomPosterize(bits=2),
                #transforms.RandomSolarize(threshold=200),
                #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),
                #transforms.CenterCrop(800),
                transforms.Resize((256, 256)), #229 for inception v3 only
                # transforms.Normalize([105.424, 105.424, 105.424], [68.886, 68.886, 68.886])
                # if use 0.485 etc, three channels now have different distributions, will hence give (RGB) COLORED image
            ]),



            'test': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # transforms.Normalize([105.424, 105.424, 105.424], [68.886, 68.886, 68.886])

         ]),

             # used for majority voting only
             'test2': transforms.Compose([
                 transforms.ToPILImage(),
                 transforms.Resize((512, 512)),
                 # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 # transforms.Normalize([105.424, 105.424, 105.424], [68.886, 68.886, 68.886])

             ])

    }

         return transform_dict1[self.phase]

    def transform_2(self, image):

        transform_dict2 = {

            'train':
                 warpim.elastic_transform(image, image.shape[1] * 2, image.shape[1] * 0.08, image.shape[1] * 0.08),
            'test': transforms.Compose([
         ])

    }

        return transform_dict2[self.phase]
 # compress to between 1 and 0 or use image's self mean and std.