# Dual-Model SAHI Detection Script

Advanced script for running two YOLO models (face + plate detection) on large image collections with SAHI (Slicing Aided Hyper Inference).

## Features

✓ **Dual Model Processing** - Run face and plate detection in a single pass
✓ **Resume Capability** - Automatically resume from where you left off if interrupted
✓ **Directory Filtering** - Process only directories containing specific text (e.g., "stitched")
✓ **Master Output File** - All detections in one YOLO-format file with full image paths
✓ **Optional Visualization** - Randomly sample N images for visual verification
✓ **Progress Tracking** - Automatic progress file to track processed images

## Output Format

Master detection file format (YOLO with full paths):
```
# image_path class_id x_center y_center width height confidence
D:\images\folder1\img1.jpg 0 0.512345 0.678901 0.123456 0.234567 0.987654
D:\images\folder1\img1.jpg 1 0.234567 0.345678 0.098765 0.123456 0.956789
D:\images\folder2\img2.jpg 0 0.445566 0.556677 0.112233 0.223344 0.934567
```

**Class IDs:**
- `0` = Face detection
- `1` = Plate detection

## Installation

```bash
pip install sahi ultralytics
```

## Usage Examples

### Using Config File (Recommended)

The easiest way to run the script is using a JSON config file:

```bash
python sahi_dual_model.py --config config.json
```

**Example config.json:**
```json
{
  "description": "Default configuration",
  "input_dir": "D:\\images\\folder",
  "output_file": "D:\\output\\detections.txt",
  "face_model": "D:\\models\\face_model.pt",
  "plate_model": "D:\\models\\plate_model.pt",
  "device": "cuda:0",
  "slice_height": 1280,
  "slice_width": 1280,
  "overlap_ratio": 0.2,
  "visualize": 0,
  "visualization_dir": null,
  "resume": true,
  "dir_filter": null
}
```

**Multiple config files for different scenarios:**
```bash
# Quick test
python sahi_dual_model.py --config config_test.json

# Only stitched directories
python sahi_dual_model.py --config config_stitched.json

# High-resolution processing
python sahi_dual_model.py --config config_highres.json
```

**Override config values via command line:**
```bash
# Use config but override specific values
python sahi_dual_model.py --config config.json --visualize 50 --device cuda:1
```

### Basic Usage (Command Line Only)

```bash
python sahi_dual_model.py \
    --input-dir "D:\PLACE - Zotac\BGD\Testing\Testing Collection Stitched" \
    --output-file "D:\PLACE - Zotac\BGD\Testing\detections_master.txt" \
    --face-model "D:\PLACE - Zotac\YOLO_DEC2025\YOLO Train\24JAN2026\runs\face1_oiv\weights\best.pt" \
    --plate-model "D:\path\to\plate_model\best.pt"
```

### With Directory Filtering

Only process directories containing "stitched":

```bash
python sahi_dual_model.py \
    --input-dir "D:\PLACE - Zotac\BGD\Testing" \
    --output-file "D:\output\detections_stitched.txt" \
    --face-model "path\to\face_model.pt" \
    --plate-model "path\to\plate_model.pt" \
    --dir-filter "stitched"
```

### With Visualization (50 Random Samples)

```bash
python sahi_dual_model.py \
    --input-dir "D:\images" \
    --output-file "D:\output\detections.txt" \
    --face-model "face_model.pt" \
    --plate-model "plate_model.pt" \
    --visualize 50 \
    --visualization-dir "D:\output\visualizations"
```

### Custom SAHI Parameters

```bash
python sahi_dual_model.py \
    --input-dir "D:\images" \
    --output-file "D:\output\detections.txt" \
    --face-model "face_model.pt" \
    --plate-model "plate_model.pt" \
    --slice-height 1280 \
    --slice-width 1280 \
    --overlap-ratio 0.3 \
    --device "cuda:0"
```

### Start Fresh (No Resume)

```bash
python sahi_dual_model.py \
    --input-dir "D:\images" \
    --output-file "D:\output\detections.txt" \
    --face-model "face_model.pt" \
    --plate-model "plate_model.pt" \
    --no-resume
```

### Combined: Filter + Visualize + Custom Parameters

```bash
python sahi_dual_model.py \
    --input-dir "D:\PLACE - Zotac\BGD\Testing" \
    --output-file "D:\output\detections_all.txt" \
    --face-model "face_model.pt" \
    --plate-model "plate_model.pt" \
    --dir-filter "stitched" \
    --visualize 100 \
    --visualization-dir "D:\output\vis" \
    --slice-height 1920 \
    --slice-width 1920 \
    --overlap-ratio 0.25 \
    --device "cuda:0"
```

## Command Line Arguments

### Config File

| Argument | Description |
|----------|-------------|
| `--config` | Path to JSON config file (when provided, all other arguments become optional and can override config values) |

### Required Arguments (optional if using config file)

| Argument | Description |
|----------|-------------|
| `--input-dir` | Root directory containing images (searched recursively) |
| `--output-file` | Path to master detection output file |
| `--face-model` | Path to face detection model weights (.pt file) |
| `--plate-model` | Path to plate detection model weights (.pt file) |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | `cuda:0` | Device for inference (cuda:0, cuda:1, cpu) |
| `--slice-height` | `1280` | Height of image slices for SAHI |
| `--slice-width` | `1280` | Width of image slices for SAHI |
| `--overlap-ratio` | `0.2` | Overlap ratio between slices (0.2 = 20%) |
| `--visualize` | `0` | Number of random images to visualize (0 = none) |
| `--visualization-dir` | `None` | Directory to save visualizations (required if visualize > 0) |
| `--no-resume` | `False` | Start fresh without resuming from progress file |
| `--dir-filter` | `None` | Only process directories containing this text |

## Resume Feature

The script automatically creates a progress file (`<output_file>_progress.txt`) that tracks processed images.

**If interrupted:**
- Simply run the same command again
- The script will skip already processed images
- Progress continues from where it left off

**To start fresh:**
- Add `--no-resume` flag, OR
- Delete the progress file manually

## Directory Filtering

Use `--dir-filter` to process only images in directories containing specific text:

```bash
--dir-filter "stitched"  # Processes: D:\images\stitched_v1\img.jpg
                         # Skips: D:\images\original\img.jpg
```

Filter is **case-insensitive** and matches any parent directory in the path.

## Visualization

When `--visualize N` is specified:
- Randomly samples N images from the collection
- Creates two visualizations per sampled image:
  - `{filename}_faces.jpg` - Face detections
  - `{filename}_plates.jpg` - Plate detections
- Useful for quality checking without processing all images visually

## Output Files

1. **Master detection file** (`--output-file`)
   - One row per detection
   - YOLO format with full image paths
   - Class 0 = face, Class 1 = plate

2. **Progress file** (`<output_file>_progress.txt`)
   - One row per processed image
   - Used for resume capability
   - Can be safely deleted to restart

3. **Visualizations** (if `--visualize > 0`)
   - Saved to `--visualization-dir`
   - Separate images for face and plate detections

## Performance Tips

1. **Slice Size**: Larger slices (1920x1920) are faster but may miss small objects
2. **Overlap**: Higher overlap (0.3-0.4) improves detection but increases processing time
3. **Visualization**: Only visualize when needed for quality checks (slows processing)
4. **Device**: Use GPU (`cuda:0`) for 10-100x faster processing than CPU

## Example Workflow

```bash
# Step 1: Edit config_test.json with your paths
# Change face_model, plate_model, input_dir, output_file

# Step 2: Test on small sample with visualization
python sahi_dual_model.py --config config_test.json

# Step 3: Review visualizations in output directory

# Step 4: Run on full dataset (use config_stitched.json or config.json)
python sahi_dual_model.py --config config_stitched.json

# Step 5: If interrupted, simply re-run the same command
# It will automatically resume from where it stopped
python sahi_dual_model.py --config config_stitched.json
```

## Troubleshooting

**Error: "Input directory does not exist"**
- Check that the path is correct
- Use absolute paths (e.g., `D:\folder` not `folder`)

**Error: "Failed to load model"**
- Verify model paths are correct
- Ensure model files (.pt) exist
- Check that models are compatible with SAHI/Ultralytics

**Slow processing:**
- Reduce slice overlap (`--overlap-ratio 0.1`)
- Increase slice size (`--slice-height 1920 --slice-width 1920`)
- Disable visualization for full runs
- Ensure using GPU (`--device cuda:0`)

**Out of memory:**
- Reduce slice size (`--slice-height 640 --slice-width 640`)
- Use CPU (`--device cpu`) if GPU memory is insufficient

## Reading the Output File

Python example to read detections:

```python
import pandas as pd

# Read detections
df = pd.read_csv('detections.txt', sep=' ', comment='#',
                 names=['image_path', 'class_id', 'x_center', 'y_center', 
                        'width', 'height', 'confidence'])

# Filter by class
faces = df[df['class_id'] == 0]
plates = df[df['class_id'] == 1]

# Group by image
detections_per_image = df.groupby('image_path').size()

# Filter by confidence
high_conf = df[df['confidence'] > 0.8]
```

## License

This script uses SAHI and Ultralytics libraries. Please comply with their respective licenses.
