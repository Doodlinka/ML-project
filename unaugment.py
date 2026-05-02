import os
from pathlib import Path
os.chdir(os.path.dirname(__file__))

# --- CONFIGURATION ---
# Set this to False to actually delete the files
DRY_RUN = False 

# Base paths for your dataset
# Adjust these if your folder names or structures are different
IMAGE_DIR = Path('./images')
LABEL_DIR = Path('./labels')

def cleanup_augmented_files():
    # We look for anything containing 'aug' in the name
    pattern = 'aug*'
    
    # 1. Find the target files
    image_files = list(IMAGE_DIR.glob(pattern))
    label_files = list(LABEL_DIR.glob(pattern))
    
    total_files = len(image_files) + len(label_files)
    
    if total_files == 0:
        print("No files found containing 'aug' in the name.")
        return

    print(f"Found {len(image_files)} images and {len(label_files)} labels to remove.")
    
    if DRY_RUN:
        print("\n--- DRY RUN: The following files WOULD be deleted ---")
        for f in image_files + label_files:
            print(f"WOULD DELETE: {f}")
        print("\n--- END OF DRY RUN ---")
        print("Set 'DRY_RUN = False' in the script to perform the actual deletion.")
    else:
        print("\nDeleting files...")
        count = 0
        for f in image_files + label_files:
            try:
                f.unlink() # This deletes the file
                count += 1
            except Exception as e:
                print(f"Error deleting {f}: {e}")
        
        print(f"Done! Successfully deleted {count} files.")

if __name__ == "__main__":
    # Check if directories exist first
    if not IMAGE_DIR.exists() or not LABEL_DIR.exists():
        print(f"Error: Could not find directories. Checked: {IMAGE_DIR} and {LABEL_DIR}")
    else:
        cleanup_augmented_files()