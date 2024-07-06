import argparse
import os
from sahi.predict import predict
import datetime


# Function to get the default model path
def default_model_path():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(script_dir, "model.pt")


# Function to get the default model configuration path
def default_config_path():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(script_dir, "config.yaml")


# Parse command line arguments
parser = argparse.ArgumentParser(description="Run YOLOv8 model for prediction.")
parser.add_argument(
    "--model_path",
    type=str,
    default=default_model_path(),
    help="Path to the YOLOv8 model weights file. Default: model.pt in script directory.",
)
parser.add_argument(
    "--model_config_path",
    type=str,
    default=default_config_path(),
    help="Path to the model configuration YAML file",
)
parser.add_argument(
    "--source", type=str, required=True, help="Source directory or file for prediction"
)
parser.add_argument(
    "--project", type=str, required=True, help="Directory to save prediction results"
)
parser.add_argument(
    "--slice_height", type=int, default=500, help="Height of each prediction slice"
)
parser.add_argument(
    "--slice_width", type=int, default=500, help="Width of each prediction slice"
)
parser.add_argument(
    "--overlap_height_ratio",
    type=float,
    default=0.2,
    help="Overlap ratio for height between slices",
)
parser.add_argument(
    "--overlap_width_ratio",
    type=float,
    default=0.2,
    help="Overlap ratio for width between slices",
)
parser.add_argument(
    "--model_device",
    type=str,
    default="0,1,2,3",
    help='Device to run the model on (e.g., "cuda:0" or "0,1,2,3")',
)
parser.add_argument(
    "--model_confidence_threshold",
    type=float,
    default=0.5,
    help="Confidence threshold for predictions",
)

args = parser.parse_args()

start = datetime.datetime.now()
print(start)

# Perform prediction using sahi.predict.predict function
result = predict(
    model_type="yolov8",
    model_path=args.model_path,
    model_config_path=args.model_config_path,
    model_device=args.model_device,
    model_confidence_threshold=args.model_confidence_threshold,
    source=args.source,
    project=args.project,
    slice_height=args.slice_height,
    slice_width=args.slice_width,
    overlap_height_ratio=args.overlap_height_ratio,
    overlap_width_ratio=args.overlap_width_ratio,
)

end = datetime.datetime.now()
print(end)
print("Inference duration:", str(end - start))
