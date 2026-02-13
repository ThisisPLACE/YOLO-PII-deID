import cv2
import os
from pathlib import Path
from PIL import Image
import random
import argparse

def load_detections_robust(file_path):
    """
    Manually parses the detection file to handle spaces and quotes in image paths.
    Assumes format: [Image Path] [Class_ID] [X_Center] [Y_Center] [Width] [Height] [Confidence]
    """
    detections_list = []
    print(f"Reading detections from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            # Skip empty lines or comments
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            
            # We expect at least 7 parts (Path + 6 numeric values)
            if len(parts) < 7:
                print(f"Warning: Skipping malformed line {i}: {line}")
                continue
            
            try:
                # The last 6 elements are always the numeric data
                conf = float(parts[-1])
                h = float(parts[-2])
                w = float(parts[-3])
                y_c = float(parts[-4])
                x_c = float(parts[-5])
                class_id = int(parts[-6])
                
                # Everything before the last 6 elements is the image path
                # Join them back with spaces and strip potential quotes
                img_path_str = " ".join(parts[:-6]).strip('"\'')
                
                detections_list.append({
                    'image_path': img_path_str,
                    'class_id': class_id,
                    'x_center': x_c,
                    'y_center': y_c,
                    'width': w,
                    'height': h,
                    'confidence': conf
                })
            except ValueError as e:
                print(f"Error parsing line {i}: {e}")
                continue
                
    return detections_list

def visualize_detections(input_file, output_dir, quality=50, num_images=None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Use the new robust loader
    detections = load_detections_robust(input_file)

    # Organize by image path
    images_dict = {}
    for d in detections:
        path = d['image_path']
        if path not in images_dict:
            images_dict[path] = []
        images_dict[path].append(d)

    # If num_images is specified, randomly sample the images
    if num_images is not None:
        sampled_keys = random.sample(list(images_dict.keys()), min(num_images, len(images_dict)))
        images_dict = {key: images_dict[key] for key in sampled_keys}

    print(f"Processing {len(images_dict)} unique images...")

    for img_path_str, dets in images_dict.items():
        img_path = Path(img_path_str)

        if not img_path.exists():
            print(f"Error: Image not found: {img_path}")
            continue

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Error: Could not read image: {img_path}")
            continue

        img_h, img_w = image.shape[:2]

        for d in dets:
            # Convert normalized to pixel coordinates
            x_px = d['x_center'] * img_w
            y_px = d['y_center'] * img_h
            w_px = d['width'] * img_w
            h_px = d['height'] * img_h

            x1 = int(x_px - (w_px / 2))
            y1 = int(y_px - (h_px / 2))
            x2 = int(x_px + (w_px / 2))
            y2 = int(y_px + (h_px / 2))

            # Class 0: Blue (Face), Class 1: Green (Plate)
            color = (255, 0, 0) if d['class_id'] == 0 else (0, 255, 0)
            label = "Face" if d['class_id'] == 0 else "Plate"

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{label} {d['confidence']:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Convert to PIL for high-quality JPG compression
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        save_path = output_path / f"{img_path.stem}_visualized.jpg"
        pil_img.save(save_path, "JPEG", quality=quality, optimize=True)
        print(f"Saved: {save_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize object detections on images.")
    parser.add_argument("--input", type=str, required=True, help="Path to the detection file.")
    parser.add_argument("--output", type=str, required=True, help="Directory to save visualized images.")
    parser.add_argument("--num-images", type=int, default=None, help="Number of random images to visualize.")
    parser.add_argument("--quality", type=int, default=50, help="JPEG quality for saved images (default: 50).")

    args = parser.parse_args()

    visualize_detections(
        input_file=args.input,
        output_dir=args.output,
        quality=args.quality,
        num_images=args.num_images
    )