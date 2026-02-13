# Visualize Detections

This script visualizes object detections (e.g., faces and plates) on images based on a detection file.

## Usage

Run the script from the command line with the following arguments:

```bash
python visualize_detections.py --input <path_to_detection_file> \
                               --output <output_directory> \
                               [--num-images <number_of_images>] \
                               [--quality <jpeg_quality>]
```

### Arguments
- `--input`: Path to the detection file (required).
- `--output`: Directory to save visualized images (required).
- `--num-images`: Number of random images to visualize (optional).
- `--quality`: JPEG quality for saved images (default: 50).

### Example

```bash
python visualize_detections.py --input C:\path\to\detections_master.txt \
                               --output output_visualizations \
                               --num-images 10 \
                               --quality 75
```

## Notes
- The script supports high-quality JPEG compression.
- Ensure all image paths in the detection file are valid.
- Install required libraries: `cv2`, `Pillow`, etc.

## Output
Visualized images will be saved in the specified output directory with bounding boxes and labels.