import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# --- 1. LOAD CUSTOM DATA WITH IMAGEFOLDER ---
# We resize everything to 128x128 before turning it into a Tensor
custom_transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Point ImageFolder to your main directory
# It will automatically find the 'gta_v' and 'rdr2' folders and label them 0 and 1
# custom_dataset = torchvision.datasets.ImageFolder(root='./archive2\Mechanical Tools Image dataset\Mechanical Tools Image dataset', transform=custom_transform)
custom_dataset = torchvision.datasets.ImageFolder(root='./archive1', transform=custom_transform)

# Print out the classes it found just to be sure
print(f"Found classes: {custom_dataset.classes}")

# Create the data loader
train_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=32, shuffle=True)


# --- 2. ADAPT THE CNN ARCHITECTURE ---
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # Layer 1: Convolutions and Pooling
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2) # Halves the image size (128 -> 64)
        
        # Layer 2: Let's add a second layer to handle the larger image better
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) # Halves the image size again (64 -> 32)
        
        # Final Linear Layer
        # The image started at 128. Pool1 made it 64. Pool2 made it 32.
        # We now have 32 channels of 32x32 images. 
        # So the flattened size is: 32 * 32 * 32
        self.fc1 = nn.Linear(32 * 32 * 32, 2) # "2" because we only have 2 classes (GTA and RDR2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Flatten the data for the final layer
        x = x.view(-1, 32 * 32 * 32) 
        x = self.fc1(x)
        return x

# You can now send this model to your GPU and run the exact same training loop!
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomCNN().to(device)
print("Ready to train on custom images!")