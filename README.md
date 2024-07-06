# YOLO and Sahi Helper Scripts

This project contains multiple Python scripts for processing images using the YOLOv8 model. Each script has a specific purpose, ranging from memory-efficient list segmentation to advanced object detection and batch inference. Below is a detailed description of each script and its usage.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/WaddahHago/YOLO-PII-deID.git
   cd YOLO-PII-deID/
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

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

- #### `yolo_squired.py`
    `python scripts/yolo_squired.py --images_dir /path/to/images --labels_dir /path/to/labels`

This script processes label files (*.txt) in a specified directory (labels_dir), performs object detection using a custom YOLO model, and saves the results in a format suitable for further analysis or processing.

    --images_dir: Path to the directory containing source images.
    --labels_dir: Path to the directory containing label files (.txt).
    --model: Path to the custom YOLO model file. Default is "model.pt" in the script's directory.

During execution, the script processes each label file in labels_dir, performs object detection using the specified YOLO model, and saves the detection results for each image.
- #### `sahi_test.py`
    `python scripts/sahi_test.py --source /path/to/directory`

This script runs predictions using the YOLOv8 model on a specified source directory or file, saving the results to the project directory. It offers customization of prediction parameters for precise control over prediction quality and speed.

    --source: Source directory or file for prediction.
    --model_path: Path to the YOLOv8 model weights file. Default is "model.pt" in the script's directory.
    --model_config_path: Path to the model configuration YAML file. Default is "config.yaml" in the script's directory.
    --project: Directory to save prediction results. Default is generated run folder in execution directory.
    --slice_height: Height of each prediction slice (default: 500).
    --slice_width: Width of each prediction slice (default: 500).
    --overlap_height_ratio: Overlap ratio for height between slices (default: 0.2).
    --overlap_width_ratio: Overlap ratio for width between slices (default: 0.2).
    --model_device: Device to run the model on (e.g., "cuda:0" or "0,1,2,3") (default: "0,1,2,3").
    --model_confidence_threshold: Confidence threshold for predictions (default: 0.5).

This script allows for fine-grained adjustment of prediction parameters to optimize the balance between prediction accuracy and computational efficiency.

- #### `sahi_predict.py`
    `python scripts/sahi_predict.py --source /path/to/directory`
This script performs batch inference on a specified source directory or file, utilizing slices of defined dimensions and overlap ratios. It logs the inference duration to provide insights into the processing time.

    --source: Source directory or file for inference.
    --model_path: Path to the YOLOv8 model weights file. Default is "model.pt" in the script's directory.
    --device: Device to run inference on (e.g., "cuda:0,1,2,3").
    --slice_height: Height of each slice for batch processing.
    --slice_width: Width of each slice for batch processing.
    --overlap_height_ratio: Overlap ratio for height between slices.
    --overlap_width_ratio: Overlap ratio for width between slices.

The script provides a detailed log of the inference duration, helping to understand and optimize the processing time for batch inferences.

- #### `list_segmenter.py`

This script iterates through a given list in batches, allowing for efficient memory usage by processing elements in segments rather than loading the entire list into memory at once.

# Contributing

Feel free to fork this project, make improvements, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

# License

This project is licensed under the MIT License - see the LICENSE file for details.
