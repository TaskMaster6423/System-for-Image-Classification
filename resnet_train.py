import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# --- 1. SET UP THE DEVICE ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 2. LOAD CUSTOM DATA ---
# IMPORTANT: ResNet expects images to be 224x224!
custom_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(), # A little data augmentation
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Specific ResNet colors
])

# ⚠️ UPDATE THIS: Put your actual dataset folder path here
dataset_path = r'./archive1' 
custom_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=custom_transform)

num_classes = len(custom_dataset.classes)
print(f"Found {num_classes} tool categories!")

train_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=32, shuffle=True)


# --- 3. LOAD RESNET18 & MODIFY IT ---
print("Downloading pre-trained ResNet18 model...")
# Download the model with its best pre-trained weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# "Freeze" all the existing layers so they don't get overwritten
for param in model.parameters():
    param.requires_grad = False

# Chop off the final layer and replace it with a new one tailored to your tools
# model.fc is the "Fully Connected" final layer of ResNet
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes) 

# Send the modified model to the GPU
model = model.to(device)


# --- 4. SET THE RULES ---
criterion = nn.CrossEntropyLoss()
# Notice we are ONLY giving the optimizer the parameters of the brand new layer (model.fc)
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)


# --- 5. THE TRAINING LOOP ---
epochs = 10 

print("Starting Transfer Learning...")
for epoch in range(epochs):
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad() 
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()       
        optimizer.step()      
        
        running_loss += loss.item()
        
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f"--- Epoch {epoch+1} completed. Average Loss: {running_loss/len(train_loader):.4f} ---")

print("Finished Training!")

# --- 6. SAVE THE SUPERCHARGED MODEL ---
torch.save(model.state_dict(), 'resnet_tools_classifier.pth')
print("Model saved successfully as 'resnet_tools_classifier.pth'!")