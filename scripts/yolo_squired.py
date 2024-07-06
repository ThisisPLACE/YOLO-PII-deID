import argparse
import logging
from ultralytics import YOLO
import os
import csv
import cv2
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Helper functions
def read_ann_file(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            x = line.strip().split()  # split the line into a list
            data.append(x)
    return data


def cv2_np_list(box, img_path):
    img = cv2.imread(img_path)
    y0, x0, _ = img.shape
    x = int(float(box[1]) * x0)
    y = int(float(box[2]) * y0)
    w = int(float(box[3]) / 2 * x0)
    h = int(float(box[4]) / 2 * y0)
    crop = img[y - h : y + h, x - w : x + w]
    return crop, y0, x0


def csv_write(path, boxes):
    with open(path, "w", newline="") as csv_f:
        writer = csv.writer(csv_f, delimiter=" ")
        for k, box in enumerate(boxes):
            writer.writerow(["0", box[0], box[1], box[2], box[3]])


def generate_unique_run_title(images_dir):
    dir_name = os.path.basename(images_dir)
    base_run_title = f"run_{dir_name}"
    suffix_num = 1
    while os.path.exists(
        os.path.join(
            os.path.dirname(images_dir), f"{base_run_title}_{suffix_num}"
        )
    ):
        suffix_num += 1
    unique_run_title = f"{base_run_title}_{suffix_num}"
    return unique_run_title


def process_labels_file(
    file, images_dir, labels_dir, model, run_title, confi, iou
):
    origin_image = os.path.join(images_dir, file.replace(".txt", ".jpg"))
    out_path = origin_image.replace(".jpg", ".txt")

    crops = read_ann_file(os.path.join(labels_dir, file))

    if os.path.exists(out_path):
        logger.debug(f"Skipping {out_path} as it already exists")
        return

    crops_np = []
    for crop in crops:
        if crop[0] in ["0", "2", "3", "5", "7"]:
            crop_np, imgw, imgh = cv2_np_list(crop, origin_image)
            crops_np.append(crop_np)
    if len(crops_np) == 0:
        logger.warning(f"No valid crops found in {file}")
        return

    results = model.predict(
        source=crops_np,
        save=True,
        imgsz=640,
        iou=iou,
        conf=confi,
        stream=True,
        device=[1],
        save_crop=False,
        save_txt=True,
        project=f"{run_title}\\crops",
        name=run_title,
    )

    xbboxs = []
    for n, result in enumerate(results):
        for c, box in enumerate(result.boxes):
            detect_box = box.xywhn.tolist()[0]
            float_list = [float(i) for i in crops[n][1:5]]
            x1, y1, w1, h1 = (
                float_list[0],
                float_list[1],
                float_list[2],
                float_list[3],
            )
            x2, y2, w2, h2 = (
                detect_box[0],
                detect_box[1],
                detect_box[2],
                detect_box[3],
            )
            xbboxs.append(
                [
                    ((x2 * w1) + (x1 - w1 / 2)),
                    ((y2 * h1) + (y1 - h1 / 2)),
                    (w1 * w2),
                    (h1 * h2),
                ]
            )

    logger.info(f"Processed {file}")

    csv_write(out_path, xbboxs)


def main(images_dir, labels_dir, model_path):
    model = YOLO(model_path)

    start = datetime.datetime.now()
    logger.info(f"Started processing at {start.strftime('%H:%M:%S')}")

    run_title = generate_unique_run_title(images_dir)
    logger.info(f"Generated unique run_title: {run_title}")

    confi = 0.01  # Confidence threshold
    iou = 0.05  # Intersection Over Union threshold

    label_files = os.listdir(labels_dir)
    for n, file in enumerate(label_files):
        process_labels_file(
            file, images_dir, labels_dir, model, run_title, confi, iou
        )
        logger.info(f"Processed {n+1}/{len(label_files)} labels files")

    end = datetime.datetime.now()
    logger.info(f"Finished processing at {end.strftime('%H:%M:%S')}")
    logger.info(f"Inference duration: {str(end - start)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO object detection script"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Path to the source images",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        required=True,
        help="Path to the source labels",
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_path = os.path.join(script_dir, "model.pt")
    parser.add_argument(
        "--model",
        type=str,
        default=default_model_path,
        help=f"Path to the custom YOLO model (default: model.pt in {script_dir})",
    )

    args = parser.parse_args()

    # Setup logging for console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    main(args.images_dir, args.labels_dir, args.model)
