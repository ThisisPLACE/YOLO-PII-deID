import argparse
from ultralytics import YOLO
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

script_dir = os.path.dirname(os.path.abspath(__file__))


def get_unique_run_title(source):
    base_run_title = f"run_{os.path.basename(source)}"
    run_title = base_run_title
    project = os.getcwd()
    counter = 1
    while os.path.exists(os.path.join(project, run_title)):
        run_title = f"{base_run_title}_{counter}"
        counter += 1
    return run_title


def create_run_directory(run_title):
    project = os.getcwd()
    run_dir = os.path.join(project, run_title)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        logging.info(f"Created directory for '{run_title}' at '{run_dir}'")
    else:
        logging.info(
            f"Directory for '{run_title}' already exists at '{run_dir}'"
        )


def resolve_path(path, default_subpath):
    project = os.getcwd()
    if path is None:
        resolved_path = os.path.join(project, default_subpath)
    elif not os.path.isabs(path):
        resolved_path = os.path.join(project, path)
    else:
        resolved_path = path

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(
            f"The specified path does not exist: {resolved_path}"
        )

    return resolved_path


def load_yolo_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"The specified model file does not exist: {model_path}"
        )

    return YOLO(model_path)


def run_object_detection(source, model_instance, run_title, project):
    logging.info(
        f"Object detection started for '{run_title}' in project directory: '{project}'"
    )

    results = model_instance.predict(
        source=source,
        model=model_instance,
        stream=True,
        save_crop=True,
        save_conf=False,
        save_txt=True,
        imgsz=6400,
        conf=0.5,
        iou=0.3,
        save=True,
        name=run_title,
        project=project,
        classes=[0, 2, 3, 5, 7],
    )

    num_images_processed = 0
    for result in results:
        num_images_processed += 1
        logging.info(f"Processed {num_images_processed} images", end="\r")

    logging.info(f"Object detection completed for '{run_title}'")


def main(source=None, model=None):
    project = os.getcwd()
    source_path = resolve_path(source, "images")
    model_path = resolve_path(
        model,
        os.path.join(
            os.path.abspath(os.path.dirname(sys.argv[0])), "model.pt"
        ),
    )

    run_title = get_unique_run_title(source_path)
    create_run_directory(run_title)

    model_instance = load_yolo_model(model_path)
    run_object_detection(source_path, model_instance, run_title, project)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help='Path to the source directory containing images. Default is "images"',
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the YOLO model file. Default is model.pt in {script_dir}.",
    )

    args = parser.parse_args()

    main(args.source, args.model)
