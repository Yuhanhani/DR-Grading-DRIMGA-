import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy
import data_augmentation
import pandas as pd
import label_fusion
import metric_classification
import metrics
import os
import sklearn


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


if torch.cuda.is_available():
    train_label = '/well/papiez/users/pea322/C_DR_Grading/Groundtruths/labels.csv'
else:
    train_label = '/Users/mirandazheng/Desktop/label_example.csv'

if torch.cuda.is_available():
    train_input = '/well/papiez/users/pea322/fusion/input_array_single_task.npy'
else:
    train_input = '/Users/mirandazheng/Desktop/label_example.csv'

if torch.cuda.is_available():
    test_label = '/well/papiez/users/pea322/C_self_test/self_test_label.csv'
else:
    test_label = '/Users/mirandazheng/Desktop/label_example.csv'

if torch.cuda.is_available():
    test_input = '/well/papiez/users/pea322/fusion_self_test/test_input_array_single_task.npy'
else:
    test_input = '/Users/mirandazheng/Desktop/label_example.csv'

batch_size = 60


# load train data ------------------------------------------------------------------------------------------------------

train_label_df = pd.read_csv(train_label)
print(type(train_label_df))

train_input_array = numpy.load(train_input)
train_input_tensor = torch.FloatTensor(train_input_array)  # [526, 16, 3]
train_input_tensor = nn.Flatten(1, -1)(train_input_tensor)  # [526, 48]

train_data = []
train_label = []
for i in range(562):

    label = train_label_df.iloc[i, 1]
    label = torch.tensor(label)

    train_data.append([train_input_tensor[i, :], label])

    if i == 0:
        label = label.item()
        label = torch.tensor([label])
        train_label = label
    else:
        label = label.item()
        label = torch.tensor([label])
        train_label = torch.cat((train_label, label), 0)

print(train_label)
print(train_label.size())
print(type(train_label)) # type: torch.Tensor

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# load test data ------------------------------------------------------------------------------------------------------

test_label_df = pd.read_csv(test_label)

test_input_array = numpy.load(test_input)
test_input_tensor = torch.FloatTensor(test_input_array)   # [49, 16, 3]
test_input_tensor = nn.Flatten(1, -1)(test_input_tensor)  # [49, 48]

test_data = []
test_label = []
for i in range(49):

    label = test_label_df.iloc[i, 1]
    label = torch.tensor(label)

    test_data.append([test_input_tensor[i, :], label])

    if i == 0:
        label = label.item()
        label = torch.tensor([label])
        test_label = label
    else:
        label = label.item()
        label = torch.tensor([label])
        test_label = torch.cat((test_label, label), 0)

print(test_label)
print(test_label.size())
print(type(test_label))  # type: torch.Tensor

test_dataloader = DataLoader(test_data, batch_size=batch_size)

# load model -----------------------------------------------------------------------------------------------------------

model = label_fusion.label_fusion()
model = model.to(device)

# define train and test loop -------------------------------------------------------------------------------------------

def train_loop(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):

        X = X.to(device)
        y = y.to(device)

        # pred, aux_output = model(X)  # for inception v3 only
        pred = model(X)

        loss = loss_fn(pred, y)
        # loss = loss_fn.forward(pred, y) for focal loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch == int(size//batch_size):
            loss, current = loss.item(), batch * batch_size
        else:
            loss, current = loss.item(), batch * len(X)

        print(f'loss:{loss:>7f} [{current:>5d}/{size:>5d}]')

def test_loop(dataloader, model, loss_fn):

    size = len(dataloader.dataset)
    total_loss = 0
    prediction = torch.empty(batch_size, 3)

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):

            X = X.to(device)
            y = y.to(device)
            # print(X.shape)
            # print(y.shape)

            pred = model(X)

            initial_pred = pred

            loss = loss_fn(pred, y)
            # loss = loss_fn.forward(pred, y) for focal loss

            if batch == int(size//batch_size):
                loss, current = loss.item(), batch * batch_size
            else:
                loss, current = loss.item(), batch * len(X)


            print(f'loss:{loss:>7f} [{current:>5d}/{size:>5d}]')


            total_loss = total_loss + loss

            if batch == 0:
                prediction = initial_pred
            else:
                prediction = torch.cat((prediction, pred), 0)

            del X, y

    return prediction, total_loss

# start to train -------------------------------------------------------------------------------------------------------

learning_rate = 0.001 # set the initial learning rate
epochs = 100
gamma = 0.9

weight = [1, 1, 3]
weight = torch.FloatTensor(weight)
weight = weight.to(device)

loss_fn = nn.CrossEntropyLoss(reduction='sum')

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


if torch.cuda.is_available():
    project_path = '/well/papiez/users/pea322/fusion_running'
    # project_path = '/users/papiez/pea322/fusion_running'
else:
    project_path = '/Users/mirandazheng/PycharmProjects/pythonProject3'

for t in range(epochs):
    print(f'Epoch {t+1}\n train loop---------------------------- ')

    model.train()
    train_loop(train_dataloader, model, loss_fn, optimizer)

    path = os.path.join(project_path, '{}_{}_{}_{}_{}.pth'.format(batch_size, learning_rate, 'adaptive', gamma, (t+1)))

    if t == epochs-1:
        torch.save(model.state_dict(), path)

    print(f'Epoch {t+1}\n test loop---------------------------- ')

    model.eval()
    prediction, total_loss = test_loop(test_dataloader, model, loss_fn)

    scheduler.step()
    print('epoch={}, learning rate={:.6f}'.format(t+1, optimizer.state_dict()['param_groups'][0]['lr']))

    Softmax_layer = nn.Softmax(dim=1)
    prediction = Softmax_layer(prediction)
    # print(prediction)
    prediction = prediction.to(device)
    print(prediction)
    y_pred = torch.argmax(prediction, dim=1)

    print(y_pred)
    prediction = prediction.detach().cpu()
    y_pred = y_pred.detach().cpu() # only move into cpu, can then be transferred into np array

    print(test_label)

    kappa_score_1 = metric_classification.quadratic_weighted_kappa(test_label, y_pred)
    macro_auc_1 = metric_classification.roc_auc_score(test_label, prediction, average="macro", multi_class='ovo')

    # kappa_score_2 = metrics.quadratic_weighted_kappa(y_true, y_pred)
    # macro_auc_2 = metrics.macro_auc(y_true, prediction) #y_pred_encoded # both are the same as above

    macro_precision = metrics.marco_precision(test_label, y_pred)
    macro_sensitivity = metrics.marco_sensitivity(test_label, y_pred)
    macro_specificity = metrics.marco_specificity(test_label, y_pred)

    print(total_loss)

    print(kappa_score_1)
    print(macro_auc_1)

    # print(kappa_score_2)
    # print(macro_auc_2)
    print(macro_precision)
    print(macro_sensitivity)
    print(macro_specificity)

    cf_matrix = sklearn.metrics.confusion_matrix(test_label, y_pred)
    print(cf_matrix)

print('Done!')