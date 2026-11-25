import torch
print(torch.__version__)
print(torch.cuda.is_available())
x = torch.randn(10, device='cuda')
print(x.sum())