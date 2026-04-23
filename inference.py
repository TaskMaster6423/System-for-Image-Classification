import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import sys

# --- 1. SETUP & PATHS ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Put the name of the picture you want to test here
test_image_path = 't2.png'

# --- 2. THE HARDCODED CLASSES (NO DATASET FOLDER NEEDED!) ---
# ⚠️ PASTE YOUR EXACT LIST HERE. Order is extremely important!
classes = ['Air Compressors Dataset', 'Angle Grinder Dataset Collection', 'Axe AI Dataset Collection', 'Ball Valve Dataset', 'Bearing Dataset', 'Caulking Gun Dataset Collection', 'Chisel Dataset', 'Clamp Dataset Collection', 'Concrete Mixer Dataset Collection', 'Construction Earmuffs Dataset', 'Corner Trowel Dataset', 'Crowbar Dataset Collection', 'Digital Caliper Dataset Collection', 'Digital Torque Wrench Dataset', 'Drill Bit AI Dataset Collection', 'Ear Plugs Dataset', 'Electric Drill AI Dataset Collection', 'Extension Cord Dataset Collection', 'Extension Spring Dataset', 'File Tool Dataset', 'Fire Extinguisher Dataset', 'First AID Kit Dataset', 'Flashlight Dataset', 'Float Tool Dataset Collection', 'Generator Dataset Collection', 'Glue Gun AI Dataset Collection', 'Hacksaw Dataset', 'Hammer AI Dataset Collection', 'Hand Saw AI Dataset Collection', 'Head Flashlight Dataset', 'Heat Gun Dataset Collection', 'Heavy Duty Vacuum Cleaner Dataset', 'Hoe Tool Dataset Collection', 'Infrared Digital Thermometer Dataset', 'Jack Hammer Dataset', 'Jack Plane Dataset', 'Jigsaw Dataset', 'Kneepads Dataset', 'LED Light Bulb Dataset', 'Ladder Dataset Collection', 'Level Tool Dataset Collection', 'Machete Dataset', 'Mallet Dataset', 'Masking Tape Roll Dataset', 'Measuring Wheel Dataset', 'Metal Nut Dataset', 'Moisture Meter Dataset', 'Nail AI Dataset Collection', 'Nail Gun Dataset Collection', 'Paint Brush Dataset Collection', 'Paint Respirator Dataset', 'Paint Roller AI','Paint Spray Gun Dataset', 'Paint Tray Dataset', 'Palm Sander Dataset', 'Plastic Bucket Dataset', 'Pliers AI Dataset Collection', 'Plunger Dataset', 'Power Socket Dataset', 'Pry Bar Dataset', 'Putty Knife Dataset Collection', 'Respirator Mask N95 Dataset', 'Rubber Boots Dataset', 'Rubber Gloves Dataset', 'Safety Glasses Dataset Collection', 'Safety Helmet Dataset Collection', 'Safety Vest Dataset', 'Screwdriver AI Dataset Collection', 'Socket Wrench Dataset', 'Soldering Iron Dataset Collection', 'Stainless Steel Washer Dataset', 'Stainless Steel Wire Brush Dataset', 'Staple Gun Dataset Collection', 'Stud Crimper Dataset', 'Stud Finder Dataset', 'Tape Measure Dataset', 'Tape Measure Dataset Collection', 'Tin Snips Dataset', 'Toolbelt Dataset', 'Traffic Cone Dataset', 'Trowel Dataset', 'Tubing Cutter Dataset', 'Utility Knife Dataset Collection', 'Utility Torch Dataset', 'Voltage Tester AI Dataset Collection', 'Welding Gloves Dataset', 'Wheel Barrow Dataset', 'Wirecutter Dataset', 'Wrench AI Dataset Collection']
num_classes = len(classes)

# --- 3. REBUILD RESNET ARCHITECTURE & LOAD WEIGHTS ---
model = resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

try:
    # Load your custom trained weights
    model.load_state_dict(torch.load('resnet_tools_classifier.pth', map_location=device))
    model.to(device)
    model.eval() # Lock the model into "testing" mode
except FileNotFoundError:
    print("Error: Could not find 'resnet_tools_classifier.pth'. Please ensure the weights file is in the same folder.")
    sys.exit()

# --- 4. PREPARE THE NEW IMAGE ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    image = Image.open(test_image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
except FileNotFoundError:
    print(f"Error: Could not find your test image '{test_image_path}'.")
    sys.exit()

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
print(f" PREDICTION RESULTS")
print(f"===========================")
print(f" I think this image is a: {predicted_label.upper()}")
print(f" Confidence: {confidence:.2f}%")
print(f"===========================\n")