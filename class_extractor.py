import os

# ---> UPDATE THIS LINE <---
# Point this to your main dataset folder that contains all the tool subfolders
dataset_path = r'.\archive1'

print(f"Scanning directory: {dataset_path}...\n")

try:
    # 1. Look inside the main folder
    # 2. Keep only the items that are actual folders (ignore stray files)
    # 3. Sort them alphabetically (exactly how PyTorch does it)
    folder_names = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    classes = sorted(folder_names)
    
    if len(classes) == 0:
        print("Uh oh, I didn't find any folders in there. Double-check your path!")
    else:
        print(f"✅ Success! Found {len(classes)} tool categories.")
        print("Copy and paste the code below directly into your inference.py file:\n")
        print("==========================================================")
        print(f"classes = {classes}")
        print("==========================================================")

except FileNotFoundError:
    print(f"❌ Error: I could not find the folder at '{dataset_path}'.")
    print("Please make sure the path is correct and try again.")