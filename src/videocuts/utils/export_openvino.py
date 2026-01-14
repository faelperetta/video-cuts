
import os
import sys
from ultralytics import YOLO

def export_model():
    model_path = "src/videocuts/models/yolov8m-face.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model {model_path}...")
    model = YOLO(model_path)
    
    print("Exporting to OpenVINO format (FP16)...")
    # Export the model
    # format='openvino', half=True for FP16 (better for Intel Arc)
    success = model.export(format='openvino', half=True)
    
    print(f"Export returned: {success}")
    
    # Ultralytics typically saves the exported model in a directory named like the original file
    # e.g. yolov8m-face_openvino_model/
    
    expected_dir = os.path.splitext(model_path)[0] + "_openvino_model"
    if os.path.exists(expected_dir):
        print(f"SUCCESS: Exported model found at {expected_dir}")
    else:
        print(f"WARNING: Expected export dir {expected_dir} not found. Check output above.")

if __name__ == "__main__":
    export_model()
