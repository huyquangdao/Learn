import torch

# indexing
x = torch.randn(3,4,5)
print(x[0])
print(x[0][0])

print(x[0][0][0])


# torch.select_index

x = torch.rand(5,4)
print(x)
indices = torch.LongTensor([0,2])
print(indices)

print(torch.index_select(x,dim=0,index=indices)) # select the 0 and 2 rows from x
print(torch.index_select(x,dim=1,index=indices)) # select the 0 and 2 columns from x

# torch.masked_select

x = torch.rand(3,4)
print(x)

mask = x.ge(0.5)
print(mask)

print(torch.masked_select(x,mask))

