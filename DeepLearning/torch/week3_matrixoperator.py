import torch

# rows concat

x_1 = torch.randn(2,5)
y_1 = torch.rand(3,5)
z_1 = torch.cat([x_1,y_1])
print(z_1)


# columns concat
x_2 = torch.randn(4,5)
y_2 = torch.rand(4,6)
z_2 = torch.cat([x_2,y_2],dim=1)
print(z_2)

# torch.view => reshape

x = torch.rand(2,3,4)
print(x)
print(x.view(2,12))
print(x.view(2,-1))


# squeeze
x = torch.zeros(2,1,2,1)
print(x)
print(x.squeeze().shape)
print(x.squeeze(1).shape)

# unsqueeze
x = torch.rand(5)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)

# Add, subtract, sum, dot product, multiply

x = torch.tensor([1,2,3,4])
y = torch.tensor([2,3,4,5])

print(x+y)
print(x-y)
print(x.sum())
print(x.dot(y))
print(x * y)

# matrix multiply
x = torch.rand(2,3)
y = torch.rand(3,2)

print(x.mm(y))

# max
print(x.max())

