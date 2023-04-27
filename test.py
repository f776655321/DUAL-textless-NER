import torch

x = torch.tensor([[1, 3, 2], [4, 5, 6], [7, 8, 9]])
print(x)

y = torch.argmax(x,dim = 1)
print(y)