import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# --- 1. SET UP THE DEVICE ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 2. LOAD CUSTOM DATA ---
custom_transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Make sure this points to your actual dataset folder!
# dataset_path = './archive2\Mechanical Tools Image dataset\Mechanical Tools Image dataset' 
dataset_path = './archive1' 
custom_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=custom_transform)

# Automatically count your tool categories
num_classes = len(custom_dataset.classes)
print(f"Found {num_classes} classes!")

# Create the data loader
train_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=32, shuffle=True)


# --- 3. ADAPT THE CNN ARCHITECTURE ---
class CustomCNN(nn.Module):
    def __init__(self, num_classes): # We pass in your total class count here
        super(CustomCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2) 
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) 
        
        # We use your dynamic 'num_classes' for the final output
        self.fc1 = nn.Linear(32 * 32 * 32, num_classes) 

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32) 
        x = self.fc1(x)
        return x

# Initialize the model and send to GPU
model = CustomCNN(num_classes=num_classes).to(device)

# --- 4. SET THE RULES (LOSS AND OPTIMIZER) ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# --- 5. THE ACTUAL TRAINING LOOP ---
epochs = 10 # We will run through the whole dataset 10 times

print("Starting training...")
for epoch in range(epochs):
    running_loss = 0.0
    
    # Loop through batches of images
    for i, (images, labels) in enumerate(train_loader):
        # Move images and labels to your GPU
        images, labels = images.to(device), labels.to(device)
        
        # 1. Clear old gradients
        optimizer.zero_grad() 
        
        # 2. Forward pass (guess)
        outputs = model(images)
        
        # 3. Calculate error (loss)
        loss = criterion(outputs, labels)
        
        # 4. Backward pass (calculate adjustments)
        loss.backward()       
        
        # 5. Update weights
        optimizer.step()      
        
        running_loss += loss.item()
        
        # Print an update every 10 batches so you know it hasn't frozen
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f"--- Epoch {epoch+1} completed. Average Loss: {running_loss/len(train_loader):.4f} ---")

print("Finished Training!")

# --- 6. SAVE THE TRAINED MODEL ---
# Save the weights to a file so you don't lose your work!
torch.save(model.state_dict(), 'my_tools_classifier.pth')
print("Model saved successfully as 'my_tools_classifier.pth'!")