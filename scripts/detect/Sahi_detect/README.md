# SAHI Detection Pipeline with Configuration File

A refactored object detection pipeline using SAHI (Sliced Aided Hyper Inference) with YOLOv8 models, now with full configuration file support.

## Overview

This project processes images in batch using a trained YOLOv8 model with sliced inference capabilities. All parameters are managed through a centralized YAML configuration file, making it easy to adjust settings without modifying code.

## Features

- **Configuration-driven**: All settings in `config.yaml`
- **Recursive image processing**: Searches nested directories
- **Folder structure preservation**: Maintains input hierarchy in output
- **Sliced inference**: Better detection of small objects with configurable tile sizes and overlap
- **Multiple export formats**: PNG, JSON, and CSV output options
- **Comprehensive logging**: File and console logging with configurable detail level
- **Error handling**: Robust error management with detailed logging
- **Memory optimization**: GPU cache clearing and memory management options

## Installation

1. Install required dependencies:
```bash
pip install pyyaml sahi torch torchvision
```

2. Install SAHI-compatible models (if not already present):
```bash
# For YOLOv8
pip install ultralytics
```

## Configuration

Edit `config.yaml` to customize behavior:

### Essential Settings

**Directories**
- `input_dir`: Root directory containing images to process
- `output_dir`: Where results will be saved
- `logs_dir`: Where log files are stored

**Model Configuration**
- `type`: Model type (e.g., 'yolov8')
- `path`: Full path to model weights (.pt file)
- `device`: GPU device ('cuda:0', 'cpu', 'mps')
- `confidence_threshold`: Detection confidence threshold (0.0-1.0)

**Slicing Parameters** (SAHI configuration)
- `slice_height`: Height of image tiles (pixels)
- `slice_width`: Width of image tiles (pixels)
- `overlap_height_ratio`: Vertical overlap between tiles (0.2 = 20%)
- `overlap_width_ratio`: Horizontal overlap between tiles
- `perform_standard_pred`: Run standard detection in addition to sliced

### Image Processing

- `supported_formats`: Image file extensions to search for
- `case_insensitive`: Search for both uppercase and lowercase extensions
- `recursive_search`: Search subdirectories
- `max_images`: Limit processing to N images (null = no limit)

### Output Options

- `export_visuals`: Save annotated images with bounding boxes
- `visual_format`: Format for exported images ('png' or 'jpg')
- `jpg_quality`: JPG quality (1-100, default 95; higher = better quality but larger file)
- `preserve_folder_structure`: Maintain input folder hierarchy
- `export_results_json`: Save detections as JSON files
- `export_results_csv`: Save detections as CSV files
- `export_results_yolo`: Save detections in YOLO format (.txt files)
- `yolo_format`: YOLO coordinate format ('normalized' for 0-1 values, 'pixel' for pixel coordinates)

### Logging

- `level`: Logging detail level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
- `log_to_file`: Enable file logging
- `log_file`: Path to log file
- `console_output`: Print logs to console

## Usage

### Basic Usage

```bash
python sahi_detection.py
```

The script will:
1. Load configuration from `config.yaml`
2. Validate settings and model file existence
3. Load the detection model
4. Recursively find all images in the input directory
5. Process each image with sliced inference
6. Save annotated images with bounding boxes
7. Log all results to console and file

### Using a Custom Config File

```python
from sahi_detection import SAHIDetectionPipeline

pipeline = SAHIDetectionPipeline('path/to/custom_config.yaml')
pipeline.run()
```

### Programmatic Usage

```python
from sahi_detection import SAHIDetectionPipeline

# Initialize pipeline with config
pipeline = SAHIDetectionPipeline('config.yaml')

# Setup (load model, create directories)
if pipeline.setup():
    # Find images
    images = pipeline.find_images()
    print(f"Found {len(images)} images")
    
    # Process them
    pipeline.process_images(images)
```

## Configuration File Example

```yaml
directories:
  input_dir: "D:\\data\\images"
  output_dir: "D:\\data\\results"
  logs_dir: "./logs"

model:
  type: "yolov8"
  path: "D:\\models\\best.pt"
  device: "cuda:0"
  confidence_threshold: 0.25

slicing:
  enabled: true
  slice_height: 1280
  slice_width: 1280
  overlap_height_ratio: 0.2
  overlap_width_ratio: 0.2
  perform_standard_pred: false

image_processing:
  supported_formats:
    - ".jpg"
    - ".jpeg"
    - ".png"
    - ".bmp"
  case_insensitive: true
  recursive_search: true
  max_images: null

output:
  export_visuals: true
  export_format: "png"
  preserve_folder_structure: true
  export_results_json: false
  export_results_csv: false

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  log_to_file: true
  log_file: "./logs/sahi_detection.log"
  console_output: true

processing:
  batch_mode: false
  num_workers: 4
  verbose: true
  skip_existing: false

advanced:
  post_processing:
    nms_threshold: 0.5
    min_area: 0
  memory:
    clear_cache: true
    optimize_memory: true
```

## JPG Export and Quality Control

The script can export visualization images in JPG format by post-processing the PNG files that SAHI exports. This is done with PIL/Pillow and provides configurable quality settings.

### Configuration

```yaml
output:
  export_visuals: true
  visual_format: "jpg"        # 'png' or 'jpg'
  jpg_quality: 95             # 1-100 (higher = better quality, larger file)
```

### How It Works

1. SAHI exports bounding box visualization as PNG (standard SAHI behavior)
2. If `visual_format: "jpg"` is set, the PNG is automatically converted to JPG
3. PNG file is deleted after conversion to save space
4. Conversion respects the `jpg_quality` setting

| Quality | Use Case | File Size |
|---------|----------|-----------|
| 50-60 | Preview/Quick review | Very small |
| 70-80 | Balanced quality/size | Small |
| 85-90 | High quality | Medium |
| 95-100 | Maximum quality | Large |

### Examples

**For smallest file size (storage optimization):**
```yaml
output:
  visual_format: "jpg"
  jpg_quality: 60
```

**For balance (default, recommended):**
```yaml
output:
  visual_format: "jpg"
  jpg_quality: 85
```

**For maximum quality:**
```yaml
output:
  visual_format: "jpg"
  jpg_quality: 95
```

**For PNG (lossless, larger files):**
```yaml
output:
  visual_format: "png"
```



## YOLO Annotation Format

The script can export detection results in YOLO format for use with YOLOv8 training pipelines.

### YOLO Format Explained

Each `.txt` file contains one line per detection with the format:
```
<class_id> <x_center> <y_center> <width> <height>
```

**Normalized Format** (recommended, default):
- All coordinates are normalized to [0, 1]
- Example: `0 0.5234 0.6789 0.1245 0.2890`

**Pixel Format**:
- Coordinates in pixel values
- Example: `0 640 720 120 280`

### Configuration

To enable YOLO export:

```yaml
output:
  export_results_yolo: true
  yolo_format: "normalized"  # or "pixel"
```

### Example Output

Input image: `dog.jpg` (640×480)
Detection: dog at center (320, 240), size 200×150 pixels

**Normalized format:**
```
0 0.5 0.5 0.3125 0.3125
```

**Pixel format:**
```
0 320 240 200 150
```

### Using Exported Annotations for Training

YOLO format is standard for YOLOv8 training. Organize your data:

```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── val/
│       └── img3.jpg
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── img2.txt
    └── val/
        └── img3.txt
```

Then create `data.yaml`:
```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 1  # number of classes
names: ['dog']  # class names
```

Train with:
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=100
```

### SAHIDetectionConfig
Loads and manages YAML configuration with validation.

**Methods:**
- `__init__(config_path)`: Load configuration file
- `get(section, key, default)`: Get configuration value
- `__getitem__(section)`: Dictionary-style access

### SAHIDetectionLogger
Sets up logging to file and console based on config.

**Methods:**
- `__init__(config)`: Initialize logger
- `get_logger()`: Return logger instance

### SAHIDetectionPipeline
Main detection pipeline orchestrating the entire workflow.

**Key Methods:**
- `setup()`: Create directories and load model
- `find_images()`: Recursively find images
- `process_images(image_files)`: Process list of images
- `run()`: Execute complete pipeline

## Performance Tips

1. **Larger Slices for Larger Objects**: Increase `slice_height` and `slice_width` for objects that span large areas

2. **Increase Overlap for Small Objects**: Higher `overlap_height_ratio` and `overlap_width_ratio` improve small object detection but reduce speed

3. **Use GPU**: Set `device: "cuda:0"` for significant speedup

4. **Batch Processing**: Enable `batch_mode: true` for parallel processing (requires more GPU memory)

5. **Skip Visuals**: Set `export_visuals: false` if you only need detection data (faster)

6. **Limit Images**: Use `max_images` when testing configurations

## Troubleshooting

### Model Not Loading
- Verify model path in config file is correct
- Ensure model file exists and is not corrupted
- Check model type matches your file format

### Out of Memory Errors
- Reduce `slice_height` and `slice_width`
- Set `batch_mode: false`
- Enable `optimize_memory: true`
- Process fewer images at once using `max_images`

### Slow Processing
- Increase `slice_height` and `slice_width` (fewer slices)
- Reduce `overlap_height_ratio` and `overlap_width_ratio`
- Use GPU instead of CPU
- Disable JSON/CSV export if not needed

### Missing Images
- Check `supported_formats` includes your file types
- Verify `recursive_search: true` if images are in subdirectories
- Check file permissions

## Logging Output

The pipeline logs important information:

```
2024-01-25 10:30:45,123 - SAHIDetection - INFO - Starting SAHI Detection Pipeline...
2024-01-25 10:30:45,124 - SAHIDetection - INFO - Configuration file: config.yaml
2024-01-25 10:30:45,234 - SAHIDetection - INFO - Loading model from D:\models\best.pt...
2024-01-25 10:30:47,456 - SAHIDetection - INFO - ✓ Model loaded successfully
2024-01-25 10:30:47,567 - SAHIDetection - INFO - Found 125 images to process
2024-01-25 10:30:47,568 - SAHIDetection - INFO - [1/125] Processing: category1/img1.jpg
2024-01-25 10:30:48,789 - SAHIDetection - INFO - ✓ Saved visuals to: D:\results\category1
...
==================================================
Processing Summary:
  Total Images: 125
  Successful: 125
  Failed: 0
==================================================
```

## Dependencies

- **Python 3.8+**
- **PyYAML**: Configuration file parsing
- **SAHI**: Sliced inference library
- **torch**: Deep learning framework
- **ultralytics**: YOLOv8 implementation
- **sahi**: Main detection library

## License

This script is provided as-is for object detection tasks using SAHI.

## Notes

- The configuration file uses YAML format for readability
- All paths can use forward slashes (/) or backslashes (\) on Windows
- Configuration is validated on startup for errors
- Logging captures all pipeline events for debugging
