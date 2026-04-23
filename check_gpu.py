import torch

# Check if CUDA (NVIDIA GPU support) is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    device_count = torch.cuda.device_count()
    # Get the name of the first GPU
    device_name = torch.cuda.get_device_name(0)
    
    print(f"Success! Found {device_count} GPU(s).")
    print(f"Primary GPU: {device_name}")
    
    # Set the device to GPU
    device = torch.device("cuda")
else:
    print("No GPU detected by PyTorch. It will default to using the CPU.")
    device = torch.device("cpu")

# Create a dummy "image" tensor (a batch of 1 image, 3 color channels, 224x224 pixels)
# and send it to the chosen device
dummy_image = torch.rand(1, 3, 224, 224).to(device)
print(f"Dummy image tensor is currently sitting on: {dummy_image.device}")