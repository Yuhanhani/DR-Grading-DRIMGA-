import torchvision.models as models
import torch
from torch import nn
from torch.utils.data import DataLoader
import MT_test_loop_with_label
import numpy
import pandas as pd
import metric_classification
import metrics
import sklearn
import MT_dataset
import data_augmentation
import csv


# prepare test data ----------------------------------------------------------------------------------------------------
pd.options.display.max_rows = 1000

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(device)

if torch.cuda.is_available():
    test_label_title = '/well/papiez/users/pea322/MT_self_test/self_test_multi_label.csv'
else:
    test_label_title = '/Users/mirandazheng/Desktop/folder/labels.csv'

if torch.cuda.is_available():
    test_img_dir = '/well/papiez/users/pea322/MT_self_test/Original_Images'
else:
    test_img_dir = '/Users/mirandazheng/Desktop/folder'

batch_size = 25

# for resize to 256^2
transform1 = data_augmentation2.data_augmentation_transform(phase='test')
test_data1 = MT_dataset.CustomImageDataset(annotations_file=test_label_title, img_dir=test_img_dir, transform_1=transform1)
test_dataloader1 = DataLoader(test_data1, batch_size=batch_size)

print(test_data1.__len__())

test_data = []
test_label = []
for i in range(test_data1.__len__()):

    image, label = test_data1.__getitem__(i)

    test_data.append([image, label])

    if i == 0:
        test_label = label
    else:
        test_label = torch.cat((test_label, label), 0)

print(test_label)
print(test_label.size())  # torch.Size([53, 2])
print(type(test_label))  # type: torch.Tensor

# for resize to 512^2
transform2 = data_augmentation2.data_augmentation_transform(phase='test2')
test_data2 = MT_dataset.CustomImageDataset(annotations_file=test_label_title, img_dir=test_img_dir, transform_1=transform2)
test_dataloader2 = DataLoader(test_data2, batch_size=batch_size)

# prepare models -------------------------------------------------------------------------------------------------------

# model1
model1 = models.resnet34(weights='ResNet34_Weights.DEFAULT', progress=False)
model1.fc = nn.Linear(512 * 1, 6)
path = '/well/papiez/users/pea322/MT_seed6/resnet34_model/6_25_0.001_adaptive_0.9_4_augTHVRJS512_L30501_13515_0.1.pth'
model1.load_state_dict(torch.load(path))

# model2
model2 = models.densenet121(weights='DenseNet121_Weights.DEFAULT', progress=True)
model2.classifier = nn.Linear(1024, 6)
path = '/well/papiez/users/pea322/MT_seed6/densenet121_model/6_25_0.001_adaptive_0.8_5_augTHVRJS256_L321_134_0.1.pth'
model2.load_state_dict(torch.load(path))

# model3
model3 = models.vgg19_bn(weights='VGG19_BN_Weights.DEFAULT', progress=False) # if no bn, gives infinity loss
model3.classifier[6] = nn.Linear(4096, 6)
path = '/well/papiez/users/pea322/MT_seed6/vgg19bn_model/6_25_0.001_adaptive_0.9_10_augTHVRJS256_L111_13515_0.1.pth'
model3.load_state_dict(torch.load(path))

# model4
model4 = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT', progress=False)
model4.classifier[1] = nn.Linear(in_features=1280, out_features=6)  # 1280, 2560
path = '/well/papiez/users/pea322/MT_seed6/efficientnetb0_model/6_25_0.001_adaptive_0.9_10_augTHVRJS256_L111_123_0.1.pth'
model4.load_state_dict(torch.load(path))

# model5
model5 = models.resnet34(weights='ResNet34_Weights.DEFAULT', progress=False)
model5.fc = nn.Linear(512 * 1, 6)
path = '/well/papiez/users/pea322/MT_seed7_new/resnet34_model/7_25_0.001_adaptive_0.9_4_augTHVRJS512_L111_123_0.01.pth'
model5.load_state_dict(torch.load(path))

# model6
model6 = models.densenet121(weights='DenseNet121_Weights.DEFAULT', progress=True)
model6.classifier = nn.Linear(1024, 6)
path = '/well/papiez/users/pea322/MT_seed7_new/densenet121_model/7_25_0.001_adaptive_0.8_3_augTHVRJS256_L111_123_0.01.pth'
model6.load_state_dict(torch.load(path))

# model7
model7 = models.densenet121(weights='DenseNet121_Weights.DEFAULT', progress=True)
model7.classifier = nn.Linear(1024, 6)
path = '/well/papiez/users/pea322/MT_seed7_new/densenet121_model/7_25_0.001_adaptive_0.8_13_augTHVRJS256_L111_123_0.pth'
model7.load_state_dict(torch.load(path))

# model8
model8 = models.vgg19_bn(weights='VGG19_BN_Weights.DEFAULT', progress=False) # if no bn, gives infinity loss
model8.classifier[6] = nn.Linear(4096, 6)
path = '/well/papiez/users/pea322/MT_seed7_new/vgg19bn_model/7_25_0.001_adaptive_0.9_20_augTHVRJS256_L111_123_0.01.pth'
model8.load_state_dict(torch.load(path))

# model9
model9 = models.vgg19_bn(weights='VGG19_BN_Weights.DEFAULT', progress=False) # if no bn, gives infinity loss
model9.classifier[6] = nn.Linear(4096, 6)
path = '/well/papiez/users/pea322/MT_seed7_new/vgg19bn_model/7_25_0.001_adaptive_0.9_11_augTHVRJS256_L111_123_0.pth'
model9.load_state_dict(torch.load(path))

# model10
model10 = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT', progress=False)
model10.classifier[1] = nn.Linear(in_features=1280, out_features=6)  # 1280, 2560
path = '/well/papiez/users/pea322/MT_seed7_new/efficientnetb0_model/7_25_0.001_adaptive_0.9_17_augTHVRJS256_L111_132_0.pth'
model10.load_state_dict(torch.load(path))

# model11
model11 = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT', progress=False)
model11.classifier[1] = nn.Linear(in_features=1280, out_features=6)  # 1280, 2560
path = '/well/papiez/users/pea322/MT_seed7_new/efficientnetb0_model/7_25_0.001_adaptive_0.9_18_augTHVRJS256_L111_132_0.01.pth'
model11.load_state_dict(torch.load(path))

# model12
model12 = models.resnet34(weights='ResNet34_Weights.DEFAULT', progress=False)
model12.fc = nn.Linear(512 * 1, 6)
path = '/well/papiez/users/pea322/MT_seed8/resnet34_model/8_25_0.001_adaptive_0.9_19_augTHVRJS512_Lrcd_02502505_1.pth'
model12.load_state_dict(torch.load(path))

# model13
model13 = models.resnet34(weights='ResNet34_Weights.DEFAULT', progress=False)
model13.fc = nn.Linear(512 * 1, 6)
path = '/well/papiez/users/pea322/MT_seed8/resnet34_model/8_25_0.001_adaptive_0.9_15_augTHVRJS512_Lrcd_02502505_2.pth'
model13.load_state_dict(torch.load(path))

# model14
model14 = models.densenet121(weights='DenseNet121_Weights.DEFAULT', progress=True)
model14.classifier = nn.Linear(1024, 6)
path = '/well/papiez/users/pea322/MT_seed8/densenet121_model/8_25_0.001_adaptive_0.8_3_augTHVRJS256_L111_133_0.1.pth'
model14.load_state_dict(torch.load(path))

# model15
model15 = models.vgg19_bn(weights='VGG19_BN_Weights.DEFAULT', progress=False) # if no bn, gives infinity loss
model15.classifier[6] = nn.Linear(4096, 6)
path = '/well/papiez/users/pea322/MT_seed8/vgg19bn_model/8_25_0.001_adaptive_0.9_10_augTHVRJS256_Lrcd_02502505_0.1.pth'
model15.load_state_dict(torch.load(path))

# model16
model16 = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT', progress=False)
model16.classifier[1] = nn.Linear(in_features=1280, out_features=6)  # 1280, 2560
path = '/well/papiez/users/pea322/MT_seed8/efficientnetb0_model/8_25_0.001_adaptive_0.9_19_augTHVRJS256_Lrcd_020404_0.1.pth'
model16.load_state_dict(torch.load(path))

# model17
model17 = models.resnext50_32x4d(weights=None, progress=True)
model17.load_state_dict(torch.load('/well/papiez/users/pea322/weights/resnext50_32x4d-7cdf4587.pth'))
model17.fc = nn.Linear(512 * 4, 6)
path = '/well/papiez/users/pea322/MT_seed6/resnext50324d_model/6_15_0.001_adaptive_0.9_4_augTHVRJS512_L111_111_0.1.pth'
model17.load_state_dict(torch.load(path))

# model18
model18 = models.wide_resnet50_2(weights=None, progress=True)
model18.load_state_dict(torch.load('/well/papiez/users/pea322/weights/wide_resnet50_2-95faca4d.pth'))
model18.fc = nn.Linear(512 * 4, 6)
path = '/well/papiez/users/pea322/MT_seed6/wide_resnet_50_2_model/6_15_0.001_adaptive_0.9_6_augTHVRJS512_L111_121_0.1.pth'
model18.load_state_dict(torch.load(path))

# model19
model19 = models.efficientnet_v2_l(weights=None, progress=False)
model19.load_state_dict(torch.load('/well/papiez/users/pea322/weights/efficientnet_v2_l-59c71312.pth'))
model19.classifier[1] = nn.Linear(in_features=1280, out_features=6)
path = '/well/papiez/users/pea322/MT_seed6/efficientnetv2l_model/6_15_0.001_adaptive_0.9_5_augTHVRJS256_L111_123_0.01.pth'
model19.load_state_dict(torch.load(path))

# model20
model20 = models.densenet161(weights='DenseNet161_Weights.DEFAULT', progress=True)
model20.classifier = nn.Linear(2208, 6) #1024, 2208, 1664, 1920
path = '/well/papiez/users/pea322/MT_seed7_new/densenet161_model/7_25_0.001_adaptive_0.8_4_augTHVRJS256_L111_123_0.01.pth'
model20.load_state_dict(torch.load(path))

# model21
model21 = models.vgg16_bn(weights=None, progress=False) # if no bn, gives infinity loss
model21.load_state_dict(torch.load('/well/papiez/users/pea322/weights/vgg16_bn-6c64b313.pth'))
model21.classifier[6] = nn.Linear(4096, 6)
path = '/well/papiez/users/pea322/MT_seed7_new/vgg16bn_model/7_25_0.001_adaptive_0.9_4_augTHVRJS256_L111_123_0.pth'
model21.load_state_dict(torch.load(path))

# model22
model22 = models.resnet152(weights='ResNet152_Weights.DEFAULT', progress=False) # equivalent to parameter = false
model22.fc = nn.Linear(512 * 4, 6)
path = '/well/papiez/users/pea322/MT_seed7_new/resnet152_model/7_25_0.001_adaptive_0.9_8_augTHVRJS256_L111_313_0.01.pth'
model22.load_state_dict(torch.load(path))

# model23
model23 = models.resnext50_32x4d(weights=None, progress=True)
model23.load_state_dict(torch.load('/well/papiez/users/pea322/weights/resnext50_32x4d-7cdf4587.pth'))
model23.fc = nn.Linear(512 * 4, 6)
path = '/well/papiez/users/pea322/MT_seed8/resnext50324d_model/8_15_0.001_adaptive_0.9_7_augTHVRJS512_Lrcd_02504503_1.pth'
model23.load_state_dict(torch.load(path))

# model24
model24 = models.densenet169(weights='DenseNet169_Weights.DEFAULT', progress=True)
model24.classifier = nn.Linear(1664, 6) #1024, 2208, 1664, 1920
path = '/well/papiez/users/pea322/MT_seed8/densenet169_model/8_25_0.001_adaptive_0.8_2_augTHVRJS256_L222_132_0.1.pth'
model24.load_state_dict(torch.load(path))

# model25
model25 = models.vgg11_bn(weights=None, progress=False) # if no bn, gives infinity loss
model25.load_state_dict(torch.load('/well/papiez/users/pea322/weights/vgg11_bn-6002323d.pth'))
model25.classifier[6] = nn.Linear(4096, 6)
path = '/well/papiez/users/pea322/MT_seed8/vgg11bn_model/8_25_0.001_adaptive_0.9_8_augTHVRJS256_L222_132_0.1.pth'
model25.load_state_dict(torch.load(path))


# model_list = [model9, model5, model6, model14] # first time
# model_list = [model16, model17, model9, model11, model18] # second time
# model_list = [model21, model22, model23, model24, model25, model26, model27]
# model_list = [model9]
#model_list = [model17, model18, model19, model20, model21, model22, model23, model24, model25]

model_list = [model1, model2, model3, model4, model22, model6, model7, model8, model9, model10, model11, model12, model13, model14, model15, model16]

length = test_data1.__len__()

# averaging-------------------------------------------------------------------------------------------------------------

prob_matrix = numpy.zeros((length, 3))

for i in range(len(model_list)):

    print(f'model_{i}')

    model = model_list[i].to(device)  # move model to GPU
    model.eval()

    if model == model1 or model == model12 or model == model13:

        prediction = MT_test_loop_with_label.test_loop(test_dataloader2, model, batch_size, device)

    else:

        prediction = MT_test_loop_with_label.test_loop(test_dataloader1, model, batch_size, device)

    #print(torch.argmax(prediction, dim=1))

    prediction = prediction.detach().cpu()
    prediction = prediction.numpy()


    print(prediction.shape)
    print(prob_matrix.shape)

    prob_matrix = prob_matrix + prediction

    #print(prediction)
    #print(prob_matrix)


del model

final_pred_prob = prob_matrix/(len(model_list))
print(final_pred_prob)
#print(numpy.sum(average_prob_matrix, axis=1))
final_prediction = numpy.argmax(final_pred_prob, axis=1)
print(final_prediction)

class_0 = 0
class_1 = 0
class_2 = 0

for i in range(length):
    if final_prediction[i] == 0:
        class_0 = class_0 + 1
    elif final_prediction[i] == 1:
        class_1 = class_1 + 1
    else:
        class_2 = class_2 + 1

print(class_0)
print(class_1)
print(class_2)

# only for with label case ---------------------------------------------------------------------------------------

test_label_medium = test_label[:, 1]  # 1D tensor
test_label = test_label_medium[test_label_medium != 3]
final_pred_prob = final_pred_prob[test_label_medium != 3]
final_prediction = final_prediction[test_label_medium != 3]
print(final_pred_prob)
# print(test_label_task3.size()) torch.Size([49])
print(final_prediction)
print(test_label)

kappa_score_1 = metric_classification.quadratic_weighted_kappa(test_label, final_prediction)
macro_auc_1 = metric_classification.roc_auc_score(test_label, final_pred_prob, average="macro", multi_class='ovo')

# kappa_score_2 = metrics.quadratic_weighted_kappa(y_true, y_pred)
# macro_auc_2 = metrics.macro_auc(y_true, prediction) #y_pred_encoded # both are the same as above

macro_precision = metrics.marco_precision(test_label, final_prediction)
macro_sensitivity = metrics.marco_sensitivity(test_label, final_prediction)
macro_specificity = metrics.marco_specificity(test_label, final_prediction)

print(kappa_score_1)
print(macro_auc_1)

# print(kappa_score_2)
# print(macro_auc_2)
print(macro_precision)
print(macro_sensitivity)
print(macro_specificity)

cf_matrix = sklearn.metrics.confusion_matrix(test_label, final_prediction)
print(cf_matrix)

# final_pred_prob = torch.FloatTensor(final_pred_prob)
# test_label = torch.FloatTensor(test_label)
# total_loss = loss_fn(final_pred_prob, test_label)
# print(total_loss)