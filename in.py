import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
from PIL import Image

# --- 1. SETUP & PATHS ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---> UPDATE THESE TWO LINES <---
# Use the exact same folder path you used in your training script
dataset_path = r'E:\archive1' 
# Type the name of the picture you want to test here
test_image_path = 't1.png' 

# --- 2. GET CATEGORY NAMES ---
try:
    dataset = datasets.ImageFolder(root=dataset_path)
    classes = dataset.classes
    num_classes = len(classes)
except FileNotFoundError:
    print(f"Error: Cannot find your dataset folder at '{dataset_path}'")
    exit()

# --- 3. REBUILD RESNET ARCHITECTURE & LOAD WEIGHTS ---
# We build a blank ResNet18 (we don't need to download the default weights this time)
model = resnet18(weights=None)

# We must modify the final layer exactly like we did in the training script
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Now we can safely load your custom trained weights!
try:
    model.load_state_dict(torch.load('resnet_tools_classifier.pth', map_location=device))
    model.to(device)
    model.eval() # Lock the model into "testing" mode
except FileNotFoundError:
    print("Error: Could not find 'resnet_tools_classifier.pth'.")
    exit()

# --- 4. PREPARE THE NEW IMAGE ---
# ResNet absolutely requires images to be resized to 224x224 and uses specific colors
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    image = Image.open(test_image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
except FileNotFoundError:
    print(f"Error: Could not find your test image at '{test_image_path}'")
    exit()

# --- 5. MAKE THE PREDICTION ---
with torch.no_grad():
    outputs = model(image_tensor)
    
    # Find the tool category with the highest score
    _, predicted_index = torch.max(outputs, 1)
    predicted_label = classes[predicted_index.item()]
    
    # Calculate how confident the model is (as a percentage)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    confidence = probabilities[predicted_index.item()] * 100

print(f"\n===========================")
print(f" RESNET PREDICTION RESULTS")
print(f"===========================")
print(f" I think this image is a: {predicted_label.upper()}")
print(f" Confidence: {confidence:.2f}%")
print(f"===========================\n")