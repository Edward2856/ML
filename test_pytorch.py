import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

a = torch.randn(10000, 10000, device=device)
b = torch.randn(10000, 10000, device=device)
# Perform matrix multiplication
c = torch.matmul(a, b)

print(torch.cuda.is_available())