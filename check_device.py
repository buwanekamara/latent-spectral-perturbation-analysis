import torch

print("====================================")
print("     CUDA DEVICE DIAGNOSTICS        ")
print("====================================")

# 1. Check if CUDA (GPU support) is available at all
cuda_available = torch.cuda.is_available()
print(f"Is CUDA (GPU) available? : {cuda_available}")

if cuda_available:
    # 2. Get the active device index and its official hardware name
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Current Active GPU Index: {current_device}")
    print(f"Active GPU Hardware Name: {device_name}")
else:
    print("No GPU detected. PyTorch is defaulting entirely to your CPU.")
print("====================================")