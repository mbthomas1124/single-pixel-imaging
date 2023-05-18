import torch

version = torch.version.cuda 
print(version)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
