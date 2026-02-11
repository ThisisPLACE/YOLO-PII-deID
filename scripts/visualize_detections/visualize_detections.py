import cv2
import pandas as pd
from pathlib import Path
from PIL import Image
import os

def visualize_detections(input_file, output_dir, quality=50):
    """
    Reads detections from a text file and draws bounding boxes on original images.
    Exports results as JPG with specified quality.
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load detections using the space-separated formats
    # Using a regex separator to handle potential extra spaces in file paths
    try:
        df = pd.read_csv(
            input_file,
            sep=r'\s+',
            engine='python',
            comment='#',
            names=['image_path', 'class_id', 'x_center', 'y_center', 'width', 'height', 'confidence']
        )
    except Exception as e:
        print(f"Error loading detection file: {e}")
        return

    # Group detections by image path to process each image once
    grouped = df.groupby('image_path')
    
    print(f"Processing {len(grouped)} images...")

    for img_path_str, detections in grouped:
        img_path = Path(img_path_str)
        
        if not img_path.exists():
            print(f"Warning: Image not found at {img_path}")
            continue

        # Load image using OpenCV
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Error: Could not read image {img_path}")
            continue

        h, w = image.shape[:2]

        # Draw each detection on the image
        for _, row in detections.iterrows():
            # Convert normalized YOLO coordinates to pixel coordinates
            # x_center, y_center, width, height are relative [0, 1]
            x_c, y_c = row['x_center'] * w, row['y_center'] * h
            bw, bh = row['width'] * w, row['height'] * h
            
            # Calculate top-left and bottom-right corners
            x1 = int(x_c - (bw / 2))
            y1 = int(y_c - (bh / 2))
            x2 = int(x_c + (bw / 2))
            y2 = int(y_c + (bh / 2))

            # Set color based on class (Blue for 0/Face, Green for 1/Plate)
            color = (255, 0, 0) if row['class_id'] == 0 else (0, 255, 0)
            label = "Face" if row['class_id'] == 0 else "Plate"
            
            # Draw rectangle and text
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{label} {row['confidence']:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Convert BGR (OpenCV) to RGB (Pillow)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        # Save as JPG with 50% quality
        save_name = f"{img_path.stem}_detected.jpg"
        save_path = output_path / save_name
        pil_img.save(save_path, "JPEG", quality=quality)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    # Update these paths to match your local setup
    INPUT_TXT = r"C:\Users\RDPUser\Desktop\PLACEPC-PROC\BGD\testing_set_300\detections_master.txt"
    OUTPUT_FOLDER = "visualized_results"
    
    visualize_detections(INPUT_TXT, OUTPUT_FOLDER, quality=50)