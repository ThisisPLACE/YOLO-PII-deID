# Create YOLO Dataset

A Python script to create properly structured YOLO training datasets from a representative sample of images with automatic train/val/test splits and a ready-to-use configuration file.

## Features

- **Recursive directory search**: Finds all images and annotation files in nested directories
- **Stratified sampling**: Ensures representative selection proportional to each folder's content
- **Automatic train/val/test splits**: Customizable ratios (default: 70/20/10)
- **YOLO-compliant structure**: Creates the exact directory layout YOLO expects
- **Automatic YAML generation**: Creates `data.yaml` ready for training
- **Class detection**: Automatically extracts class IDs from annotations
- **Reproducible results**: Optional seed for consistent runs

## Installation

No special dependencies beyond Python 3.6+, except for PyYAML:

```bash
pip install pyyaml
```

Or if you have pip installed:
```bash
pip install -r requirements.txt
```

## Usage

### Basic usage (200 images, default 70/20/10 split):
```bash
python create_yolo_dataset.py -i /path/to/images -o /path/to/dataset
```

### Custom number of images:
```bash
python create_yolo_dataset.py -i /path/to/images -o /path/to/dataset -n 500
```

### Custom train/val/test split:
```bash
python create_yolo_dataset.py -i /path/to/images -o /path/to/dataset \
    -n 300 --train 0.8 --val 0.15 --test 0.05
```

### Reproducible results with seed:
```bash
python create_yolo_dataset.py -i /path/to/images -o /path/to/dataset \
    -n 200 --seed 42
```

## Command Line Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--input` | `-i` | path | required | Root directory containing images in subdirectories |
| `--output` | `-o` | path | required | Output directory for the YOLO dataset |
| `--number` | `-n` | int | 200 | Number of images to sample |
| `--train` | - | float | 0.7 | Training set ratio (0-1) |
| `--val` | - | float | 0.2 | Validation set ratio (0-1) |
| `--test` | - | float | 0.1 | Test set ratio (0-1) |
| `--seed` | - | int | None | Random seed for reproducible sampling |

## Output Structure

The script creates the following directory structure:

```
dataset/
├── images/
│   ├── train/          (70% of images)
│   ├── val/            (20% of images)
│   └── test/           (10% of images)
├── labels/
│   ├── train/          (70% of annotations)
│   ├── val/            (20% of annotations)
│   └── test/           (10% of annotations)
└── data.yaml           (YOLO configuration file)
```

### Example data.yaml

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
test: images/test
nc: 5
names:
  - person
  - car
  - dog
  - cat
  - bicycle
```

## Training with YOLOv8

Once your dataset is created, you can train with YOLOv8 immediately:

### Python API:
```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov8n.pt')  # nano model

# Train the model
results = model.train(
    data='/path/to/dataset/data.yaml',
    epochs=100,
    imgsz=640,
    device=0  # GPU device index
)

# Run inference
results = model.predict(source='image.jpg')
```

### Command line:
```bash
yolo detect train data=/path/to/dataset/data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

## How It Works

1. **Discovery Phase**
   - Recursively walks through all subdirectories
   - Finds image files (jpg, jpeg, png, bmp, gif, tiff)
   - Matches each image with its corresponding .txt annotation file
   - Skips images without annotations

2. **Stratified Sampling Phase**
   - Groups images by source directory
   - Calculates proportion of images in each directory
   - Allocates samples proportionally to maintain representation
   - Randomly selects the calculated number from each directory

3. **Split Phase**
   - Shuffles the selected samples
   - Divides into train/val/test sets according to specified ratios

4. **Organization Phase**
   - Creates the YOLO directory structure
   - Copies images to `images/{split}/` folders
   - Copies annotations to `labels/{split}/` folders
   - Handles filename conflicts automatically

5. **Configuration Phase**
   - Extracts unique class IDs from annotations
   - Creates data.yaml with paths and class names
   - Generates summary statistics

## YOLO Annotation Format

The script expects standard YOLO format annotations:

**File naming**: Each image must have a corresponding `.txt` file with the same name
```
image.jpg  →  image.txt
photo.png  →  photo.txt
```

**Content format**: Each line contains one object
```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: Integer class identifier (0, 1, 2, ...)
- `x_center`, `y_center`: Normalized center coordinates (0-1)
- `width`, `height`: Normalized bounding box dimensions (0-1)

**Example image.txt**:
```
0 0.5 0.5 0.3 0.4
2 0.7 0.3 0.2 0.25
1 0.2 0.8 0.15 0.2
```

This represents 3 objects:
- Class 0 at center (0.5, 0.5) with width 0.3, height 0.4
- Class 2 at center (0.7, 0.3) with width 0.2, height 0.25
- Class 1 at center (0.2, 0.8) with width 0.15, height 0.2

## Examples

### Example 1: Create a dataset with 300 images (default split)
```bash
python create_yolo_dataset.py -i ~/datasets/raw_images -o ~/datasets/yolo_data -n 300
```

Output:
```
Searching for images in: ~/datasets/raw_images
Found 5000 images with annotations across 12 directories

Sampling 300 representative images...

Selected 150/800 images from: ~/datasets/raw_images/set_a
Selected 80/400 images from: ~/datasets/raw_images/set_b
...

Creating YOLO dataset structure at: ~/datasets/yolo_data
Copied 210 train images
Copied 60 val images
Copied 30 test images

Created data.yaml configuration file
  Classes: 5
  Train images: 210
  Val images: 60
  Test images: 30

============================================================
YOLO DATASET CREATED SUCCESSFULLY
============================================================
Output directory: ~/datasets/yolo_data

Dataset splits:
  Train: 210 images (70.0%)
  Val:   60 images (20.0%)
  Test:  30 images (10.0%)
  Total: 300 images
...
```

### Example 2: Create imbalanced split for research
```bash
python create_yolo_dataset.py \
    -i /data/images \
    -o /data/yolo_dataset \
    -n 1000 \
    --train 0.6 \
    --val 0.3 \
    --test 0.1 \
    --seed 42
```

## Performance Notes

- **Scalability**: Handles 10,000+ images efficiently
- **Sampling time**: < 1 second for most datasets
- **Copy time**: Depends on image size and disk speed (typically 1-5 seconds for hundreds of images)
- **Memory usage**: Minimal, O(n) where n is number of images found

## Troubleshooting

### "No images with annotation files found"
- Check that image files have corresponding .txt files with the same name
- Ensure annotation files are in the same directory as images

### "Requested N images but only M available"
- You have fewer total images than requested
- The script will use all available images and notify you

### Filename conflicts in output
- If multiple source directories have files with the same name, the script automatically appends counters
- This preserves all files and ensures no data loss

### Classes not detected correctly
- Verify annotation format: `<class_id> <x> <y> <w> <h>`
- Class IDs should be integers starting from 0

## Tips

1. **Use a seed for reproducibility**: `--seed 42` ensures you get the same split every time
2. **Start small**: Test with `--number 100` first to verify your data structure
3. **Check data.yaml**: Always verify the generated `data.yaml` before training
4. **Balanced splits**: For small datasets, consider `--train 0.7 --val 0.2 --test 0.1`
5. **Large datasets**: For 10,000+ images, consider `--train 0.8 --val 0.1 --test 0.1` for more training data

## Requirements

- Python 3.6+
- PyYAML: `pip install pyyaml`

## License

This script is provided as-is for YOLO dataset preparation.
