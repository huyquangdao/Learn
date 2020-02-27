import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn

x = torch.FloatTensor(torch.arange(1.,101.))
y = torch.linspace(start=0,end=10,steps=100) + torch.randn(100)

x = x.unsqueeze(1)
y = y.unsqueeze(1)

model = nn.Linear(1,1)
print([p for p in model.parameters()])
criterion = nn.MSELoss()
lr = 0.0001
epoch = 20

plt.ion()

for i in range(epoch):
    model.zero_grad()
    pred = model(x)
    loss = criterion(pred,y)

    plt.cla()
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), pred.data.numpy(), 'r-', lw=5)
    plt.title('loss: {}'.format(loss.item()))
    plt.pause(0.2)

    if loss.item() < 1e-3:
        break

    loss.backward()
    for param in model.parameters():
        param.data = param.data - lr * param.grad.data

plt.ioff()
plt.show()
