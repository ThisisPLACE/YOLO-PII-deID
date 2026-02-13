# Blur Anonymizer

Automatically blurs regions in images based on detection coordinates from a CSV file.

## Quick Start

### Install Dependencies
```bash
pip install opencv-python piexif
```

### Basic Usage
```bash
python anonymizer.py <csv_file> <output_dir> [parent_dir]
```

### Examples

**With parent directory:**
```bash
python anonymizer.py sample_input_file.txt ./output C:\Images\Dataset
```

**Without parent directory (uses CSV paths as-is):**
```bash
python anonymizer.py sample_input_file.txt ./output
```

## How It Works

1. Reads detections from a CSV file
2. Groups detections by image file
3. Blurs each detected region (40x40 pixel kernel)
4. Saves blurred images to output directory
5. Preserves original directory structure
6. Preserves EXIF metadata (GPS, timestamps, etc.)
7. Generates a `processing_log.txt` in output directory

## CSV Format

The CSV file must have this header:
```csv
#,image_path,class_id,x_center,y_center,width,height,confidence
```

Example:
```csv
1,\path\to\file1.jpg,0,0.656967,0.807094,0.062510,0.060177,0.721478
2,\path\to\file2.jpg,0,0.811487,0.569905,0.012730,0.025151,0.516127
```

**Coordinate Format:** YOLO normalized (0.0-1.0 range, relative to image size)

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `csv_file` | Yes | Path to CSV file with detections |
| `output_dir` | Yes | Directory for blurred images |
| `parent_dir` | No | Parent directory to prepend to image paths |

## Output

- **Blurred images:** Saved in `output_dir` with original structure preserved
- **Log file:** `processing_log.txt` in output directory with timestamps and details

## Example Output

```
✓ Successfully processed: 3 images
  Total detections blurred: 5
✗ Failed to process: 0 images
```

The `processing_log.txt` contains the same information with timestamps for each action.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Image not found" | Check parent_dir and image paths in CSV |
| "Could not load image" | Verify image file is valid and not corrupted |
| "Could not preserve EXIF" | Normal if original image has no EXIF data |

