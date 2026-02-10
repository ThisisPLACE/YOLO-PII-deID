import os
from ultralytics import YOLO

def main():
    # Use absolute paths to avoid the Linux path errors we discussed
    ROOT = "/home/user/Downloads/training_24jan"
    
    # Path to your best OIV weights
    model_path = os.path.join(ROOT, "runs/detect/train_oiv/weights/best.pt")
    data_yaml = os.path.join(ROOT, "face_set/data.yaml")
    cfg_yaml = os.path.join(ROOT, "finetune_360.yaml")

    # Load the model
    model = YOLO(model_path)

    # Execute Second Training
    model.train(
        data=data_yaml,
        cfg=cfg_yaml,         # Use our new specialized settings
        epochs=100,           # Total epochs (model will continue from current state)
        imgsz=640,            # Moderate res for faster training
        batch=-1,             # auto-batching to maximize your 6GB VRAM
        workers=4,            # Optimized for multiprocessing
        device=0,             # Explicitly target your GPU
        patience=30,          # Stop early if mAP stops improving
        close_mosaic=20,      # Longer clean-up phase to reduce False Positives
        exist_ok=True         # Save in existing directory if preferred
    )

if __name__ == '__main__':
    main()