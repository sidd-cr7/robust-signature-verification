import os
import shutil

# CHANGE THIS PATH to your extracted dataset folder
source_path = r"C:\Users\l\OneDrive\Documents\PROJECTS\extract"   

dest_real = r"dataset\real"
dest_forged = r"dataset\forged"

os.makedirs(dest_real, exist_ok=True)
os.makedirs(dest_forged, exist_ok=True)

for folder in os.listdir(source_path):
    folder_path = os.path.join(source_path, folder)

    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            if folder.endswith("_forg"):
                shutil.copy(file_path, dest_forged)
            else:
                shutil.copy(file_path, dest_real)

print("Done! Dataset organized.")