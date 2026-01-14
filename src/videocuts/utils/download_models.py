
import os
import requests
import sys

# Standard YOLO-Face model URLs (using a reliable source)
# These are typically available from the ultralytics/assets or similar trusted repos.
# Since official YOLOv8-face models aren't in the main repo, we'll use a known working release
# or the same source the user likely got the nano model from.
# For this task, we will try to find a direct link, or use the ultralytics auto-download if possible.
# Ultralytics normally downloads general models automatically, but face models are custom.

# However, since the user already has yolov8n-face.pt, we can assume a similar naming convention.
# Let's try to download from a common mirror if 'ultralytics' doesn't handle it.
# Actually, the safest bet is to rely on Ultralytics to download it if we initialize it, 
# BUT face models are often custom trained.
# Let's try to download from the popular 'akanametov/yolo-face' or similar if we can't find it.

# URL for YOLOv8m-face.pt (This is a placehoder for the actual model URL)
# We will use a known reliable source for yolov8-face models. 
# Since I cannot browse the web freely for a new URL in this specific turn without search, 
# I will use a direct download utility that tries to fetch it.

# NOTE: If we cannot find a valid URL, we might need to ask the user or use a 'general' s/m model 
# and see if it works (it won't work well for faces specifically).
# Assuming the user is using the standard implementation that often pairs with these projects:
MODEL_URL = "https://huggingface.co/deepghs/yolo-face/resolve/main/yolov8m-face/model.pt"

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False

def main():
    dest_dir = "src/videocuts/models"
    os.makedirs(dest_dir, exist_ok=True)
    
    model_name = "yolov8m-face.pt"
    dest_path = os.path.join(dest_dir, model_name)
    
    if os.path.exists(dest_path):
        print(f"Model {model_name} already exists at {dest_path}")
        return

    if not download_file(MODEL_URL, dest_path):
        print("Failed to download model.")
        sys.exit(1)

if __name__ == "__main__":
    main()
