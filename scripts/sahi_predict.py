import argparse
import os
from sahi import AutoDetectionModel
from sahi.predict import predict
import datetime


def main(
    model_path,
    device,
    source,
    slice_height,
    slice_width,
    overlap_height_ratio,
    overlap_width_ratio,
):
    start = datetime.datetime.now()

    AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=0.5,
        device=device,
    )

    # batch predict
    predict(
        model_type="yolov8",
        model_path=model_path,
        model_device=device,
        model_confidence_threshold=0.5,
        source=source,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    end = datetime.datetime.now()
    print("inference duration= ", str(end - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 inference with specified parameters."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to YOLOv8 model weights (default: model.pt in script directory)",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device to run inference on (e.g., 'cuda:0,1,2,3')",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source directory or file for inference",
    )
    parser.add_argument(
        "--slice_height",
        type=int,
        required=True,
        help="Height of each slice for batch processing",
    )
    parser.add_argument(
        "--slice_width",
        type=int,
        required=True,
        help="Width of each slice for batch processing",
    )
    parser.add_argument(
        "--overlap_height_ratio",
        type=float,
        required=True,
        help="Overlap height ratio for batch processing",
    )
    parser.add_argument(
        "--overlap_width_ratio",
        type=float,
        required=True,
        help="Overlap width ratio for batch processing",
    )

    args = parser.parse_args()

    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_path = os.path.join(script_dir, "model.pt")

    if args.model_path is None:
        model_path = default_model_path
    else:
        model_path = args.model_path

    main(
        model_path,
        args.device,
        args.source,
        args.slice_height,
        args.slice_width,
        args.overlap_height_ratio,
        args.overlap_width_ratio,
    )
