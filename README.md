# Documentation for detection and blurring faces and number plates using YOLO

This project contains multiple Python scripts for processing images using the YOLOv8 model. Each script has a specific purpose, ranging from memory-efficient list segmentation to advanced object detection and batch inference. Below is a detailed description of each script and its usage.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/WaddahHago/YOLO-PII-deID.git
   cd YOLO-PII-deID/
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Pre-trained Models


## Usage

- #### `yolo_detect.py`
    `python scripts/yolo_detect.py --source /path/to/images --model /path/to/model.pt`
  
This script performs object detection using YOLOv8 on images located in a specified source directory. It accepts two optional arguments:

    --source: Specifies the directory containing images to be processed. Default is "images" in the script's directory.
    --model: Specifies the path to the YOLOv8 model file. Default is "model.pt" in the script's directory.

During execution, the script creates a run directory in the current execution directory where it saves:

    Detected objects.
    Labels associated with detections.
    Cropped images of detected objects.

- #### `yolo_squared.py`
    `python scripts/yolo_squired.py --images_dir /path/to/images --labels_dir /path/to/labels`

This script processes label files (*.txt) in a specified directory (labels_dir), performs object detection using a custom YOLO model, and saves the results in a format suitable for further analysis or processing.

    --images_dir: Path to the directory containing source images.
    --labels_dir: Path to the directory containing label files (.txt).
    --model: Path to the custom YOLO model file. Default is "model.pt" in the script's directory.

During execution, the script processes each label file in labels_dir, performs object detection using the specified YOLO model, and saves the detection results for each image.
- #### `blur_code.py`
    `python scripts/sahi_test.py --source /path/to/directory`


# Contributing

Feel free to fork this project, make improvements, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

# License

This project is licensed under the MIT License - see the LICENSE file for details.
