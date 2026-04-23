import os
from PIL import Image

# This path is pulled directly from your error message
dataset_path = r'E:\image project\archive1'

removed_count = 0
print("Scanning dataset for corrupted images. This might take a minute...")

# Walk through every folder and file in your dataset
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)
        
        try:
            # Try to open the file and verify its integrity
            with Image.open(file_path) as img:
                img.verify()
        except Exception as e:
            # If it throws an error, it's a bad file. Delete it.
            print(f"Removing bad file: {file_path}")
            os.remove(file_path)
            removed_count += 1

print(f"\nCleanup Complete! Successfully removed {removed_count} corrupted files.")
print("You are now safe to run your training script again.")