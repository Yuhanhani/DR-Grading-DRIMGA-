# test loop without label just to return prediction results

import torch
from torch import nn

def test_loop(dataloader, model, batch_size, device): # delete loss_fn for without label case

    prediction = torch.empty(batch_size, 6)

    with torch.no_grad():

        for batch, (X, y) in enumerate(dataloader):  # change to X only in without label case

            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            initial_pred = pred

            if batch == 0:
                prediction = initial_pred
            else:
                prediction = torch.cat((prediction, pred), 0)

            print(prediction.size())

            del X

    Softmax_layer = nn.Softmax(dim=1)
    prediction = Softmax_layer(prediction[:, 3:])
    # print(f'{prediction.size()}--------')

    return prediction