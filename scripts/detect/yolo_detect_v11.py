from ultralytics import YOLO
from pathlib import Path
import os

# Configuration
src = Path(r"D:\PLACE - Zotac\BGD\Testing\Testing Collection stitched Staged 300")
project = Path(r"D:\PLACE - Zotac\BGD\Testing\test_detect\yolo11s_28JAN")
run_title = 'yolo11s_28JAN'

# Image extensions to search for
IMAGE_EXTENSIONS = {'.JPG', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# model
model = YOLO(r"D:\PLACE - Zotac\YOLO_DEC2025\YOLO Train\BGD_28JAN2026\trained_pt\train_26s_164epoch\weights\best.pt")

# Detection parameters
DETECTION_PARAMS = {
    'device': [0],
    'imgsz': 1920,
    'conf': 0.25,
    'iou': 0.1,
    'verbose': True,
    'save_crop': True,
    'classes':[0,1,2,3,5,7]  # Example: only detect persons,bicycle, cars,motorcycle, buses, truck
}

# Find all images recursively in directories containing "stitched"
image_files = []
for ext in IMAGE_EXTENSIONS:
    image_files.extend(src.rglob(f'*{ext}'))
    image_files.extend(src.rglob(f'*{ext.upper()}'))

# Filter to only include images whose parent directory path contains "stitched"
image_files = [img for img in image_files if 'stitched' in str(img.parent).lower()]
image_files = list(set(image_files))  # Remove duplicates
image_files.sort()

print(f"Found {len(image_files)} images to process")

# Process each image
for idx, img_path in enumerate(image_files):
    # Calculate relative path from source root
    rel_path = img_path.relative_to(src)
    
    # Create corresponding output directory
    output_dir = project / rel_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output text file path
    txt_path = output_dir / f"{img_path.stem}.txt"
    
    # Skip if already processed
    if txt_path.exists():
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(image_files)} images")
        continue
    
    # Run detection
    results = model.predict(
        source=str(img_path),
        **DETECTION_PARAMS
    )
    
    # Save results as text file
    result = results[0]
    with open(txt_path, 'w') as f:
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                # Format: class_id confidence x_center y_center width height (YOLO format)
                cls_id = int(box.cls)
                conf = float(box.conf)
                # Normalized coordinates
                x_norm = float(box.xywhn[0][0])
                y_norm = float(box.xywhn[0][1])
                w_norm = float(box.xywhn[0][2])
                h_norm = float(box.xywhn[0][3])
                
                f.write(f"{cls_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f} {conf:.4f}\n")
    
    # Save annotated image
    annotated_img_path = output_dir / f"{img_path.stem}_annotated.jpg"
    result.save(filename=str(annotated_img_path))
    
    # Progress tracking
    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1}/{len(image_files)} images")

print(f"Detection complete! Results saved to {project}")