import torch

print("torch", torch.__version__)
print("build cuda:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())
