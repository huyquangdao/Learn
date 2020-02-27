import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as f

x = Variable(torch.FloatTensor([1,2,3]), requires_grad = True)

print(x.data)

y = Variable(torch.FloatTensor([3,4,5]),requires_grad = True)

z = x + y
print(z.data)

s = z.sum()

s.backward()
print(x.grad)

# Variables vs parameters

x = torch.Tensor([1.,2.,3.])
vx = Variable(x)

print(x.requires_grad)

nx = nn.Parameter(x)
print(nx.requires_grad)


# torch.nn.Linear
linear = torch.nn.Linear(5,2)
data = Variable(torch.randn(2,5))
print(linear(data))

# non linearities
x = Variable(torch.tensor([1.,2.,3.]))
print(f.relu(x))
print(f.sigmoid(x))
print(f.tanh(x))

# softmax

data = Variable(torch.randn(5))
print(data)
print(f.softmax(data))
print(f.softmax(data).sum())
print(f.log_softmax(data))

