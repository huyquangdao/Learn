import torch
import numpy as np
#  Create Tensor
v_data = [[1.,2.,3.]] # shape [1,3]
v_tensor = torch.Tensor(v_data)
print(v_tensor.shape)

m_data = [[1.,2.,3.],[4.,5.,6.]] # shape [2,3]
m_tensor = torch.tensor(m_data)
print(m_tensor.shape)

t_data = [[[1,2,],[4,5,]],[[3,4,],[6,7,]]]
t_tensor = torch.tensor(t_data)
print(t_tensor.shape)

# convert tensor to list
print(t_tensor.tolist())

# Create tensor from ndarray
v_data = np.array([1,2,3])
v_tensor = torch.tensor(v_data)
print(v_tensor)

m_data = np.array([[2,3,4],[5,6,7]])
m_tensor = torch.tensor(m_data)
print(m_tensor)

t_data = np.array([[[2,3,4],[5,6,7]],[[7,8,9],[10,11,12]]])
t_tensor = torch.tensor(t_data)
print(t_tensor.shape)


# convert tensor to numpy array
print(t_tensor.numpy())

# create special tensors

x = torch.zeros(2,3)
print(x)

x = torch.ones(2,3)
print(x)

x = torch.rand(3,4)
print(x)

x = torch.randperm(5)
print(x)




