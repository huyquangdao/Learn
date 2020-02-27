from torch.utils.data import Dataset
import pandas as pd
from torch.autograd import Variable
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
df = pd.read_csv('data/logstic_data.csv',names=['fea1','fea2','label'])
x = df[['fea1','fea2']].values

x = (x - x.mean())/x.std()
y = df['label'].values

x_tensor = torch.FloatTensor(x)
y_tensor = torch.FloatTensor(y).unsqueeze(1)

epoch = 30
lr = 0.015

mask1 = y == 1.0
mask0 = y != 1.0

x_cl0 = x[mask0]
x_cl1 = x[mask1]

print(x_cl0.shape)

plt.scatter(x=x_cl0[:, 0], y=x_cl0[:, 1], c='red')
plt.scatter(x=x_cl1[:, 0], y=x_cl1[:, 1], c='blue')

plt.show()

plt.ion()

model = nn.Linear(2,1)
criterion = nn.BCEWithLogitsLoss()

for i in range(epoch):

    logit = model(x_tensor)
    loss = criterion(logit,y_tensor)

    loss.backward()
    pred_data = torch.sigmoid(logit).data.numpy()

    mask1 = pred_data >= 0.5
    mask0 = pred_data < 0.5

    x_cl0 = x[np.squeeze(mask0,axis=-1)]
    x_cl1 = x[np.squeeze(mask1,axis=-1)]

    plt.cla()
    plt.scatter(x=x_cl0[:,0],y=x_cl0[:,1],c='red')
    plt.scatter(x=x_cl1[:,0],y=x_cl1[:,1],c='blue')

    plt.title('loss: {}'.format(loss.item()))
    plt.pause(0.2)

    for param in model.parameters():
        param.data = param.data - param.grad.data


plt.ioff()
plt.show()


