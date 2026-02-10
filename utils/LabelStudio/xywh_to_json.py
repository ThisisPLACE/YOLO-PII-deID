import json
import os
from pathlib import Path

def yolo_to_label_studio(yolo_folder, images_folder, output_file):
    tasks = []
    task_id = 1
    
    for txt_file in Path(yolo_folder).glob("*.txt"):
        img_name = txt_file.stem + ".jpg"  # or .png
        img_path = os.path.join(images_folder, img_name)
        
        if not os.path.exists(img_path):
            continue
        
        results = []
        
        with open(txt_file) as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert normalized YOLO to percentage (0-100)
                x = (x_center - width/2) * 100
                y = (y_center - height/2) * 100
                w = width * 100
                h = height * 100
                
                label = "face" if class_id == 0 else "license_plate"
                
                results.append({
                    "value": {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "rotation": 0,
                        "rectanglelabels": [label]
                    },
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "id": f"result_{task_id}_{line_idx}"
                })
        
        task = {
            "id": task_id,
            "data": {
                "image": f"/data/local-files/?d=Mirpur/images/{img_name}"
            },
            "annotations": [
                {
                    "id": task_id,
                    "completed_by": 1,
                    "result": results
                }
            ]
        }
        
        tasks.append(task)
        task_id += 1
    
    with open(output_file, 'w') as f:
        json.dump(tasks, f, indent=2)
    
    print(f"Converted {len(tasks)} annotations")

# Usage
yolo_to_label_studio(
    r"D:\PLACE - Zotac\BGD\YOLO Train\training_data_27DEC2025_Mirpur\sampled\labels",  # Folder with .txt files
    r"D:\PLACE - Zotac\BGD\YOLO Train\training_data_27DEC2025_Mirpur\sampled\Mirpur\images",  # Folder with images
    "label_studio_import.json"
)