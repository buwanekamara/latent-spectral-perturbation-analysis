import shutil
from pathlib import Path

def flatten_folders(source_dir: str, target_dir: str):
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create the target directory if it doesn't exist yet
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Count how many subfolders are in the source directory
    # (This looks for directories specifically, ignoring files at the top level)
    subfolders = [f for f in source_path.iterdir() if f.is_dir()]
    print(f"Found {len(subfolders)} subfolders in '{source_dir}'.\n")
    
    if len(subfolders) == 0:
        print("No subfolders to process.")
        return

    # 2. Go inside each folder one by one
    for folder in subfolders:
        print(f"Processing folder: {folder.name}")
        
        # Iterate through all items inside this specific subfolder
        for item in folder.iterdir():
            # Check if it's a file (skips nested subfolders)
            if item.is_file():
                destination = target_path / item.name
                
                # Handle potential filename collisions
                if destination.exists():
                    print(f"  [Warning] {item.name} already exists in destination. Appending folder name to prevent overwrite.")
                    destination = target_path / f"{folder.name}_{item.name}"
                
                # 3. Move the data to the separate location
                shutil.move(str(item), str(destination))
                print(f"  Moved: {item.name} -> {destination}")
                
    print("\nAll files have been successfully consolidated!")

# --- HOW TO USE IT ---
# Replace these paths with your actual folder locations
SOURCE_FOLDER = "adm"
TARGET_FOLDER = "allfake"

flatten_folders(SOURCE_FOLDER, TARGET_FOLDER)