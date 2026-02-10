from ultralytics import YOLO

def train_street_view():
    # Load model pretrained on Open Images (already knows plates/faces)
    model = YOLO("yolov8m-oiv7.pt")

    model.train(
        data=r"D:\PLACE - Zotac\BGD\YOLO Train\training_data_31DEC\dataset\data.yaml",
        epochs=200,
        imgsz=640,           # High-res to keep small objects from disappearing
        batch=-1,             # Auto-batch (finds the max your GPU can handle)
        multi_scale=True,     # CRITICAL: Mixes your low-res and high-res data safely
        augment=True,
        # Hyperparameters for Small Object Recall:
        mosaic=1.0,           # Mixes 4 images; helps find objects in context
        copy_paste=0.3,       # Pastes plates/faces onto other backgrounds
        degrees=10.0,         # Handles skewed plate angles
        scale=0.9,            # Varies object size by 90%
        close_mosaic=10,      # Disable mosaic in last 10 epochs for stability
        project="street_view_privacy",
        name="mv11_oiv7"
    )

if __name__ == "__main__":
    train_street_view()