import torch

print(torch.cuda.is_available())
print(torch.version.cuda)            # Should match +cuXXX
print(torch.cuda.get_device_name(0)) # GPU name

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX