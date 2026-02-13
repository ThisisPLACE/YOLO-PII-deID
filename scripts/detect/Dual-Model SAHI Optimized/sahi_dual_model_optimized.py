import os
import argparse
import random
import json
import threading
import queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Set, List, Tuple, Dict
import logging
from dataclasses import dataclass
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import torch
import multiprocessing as mp
import cv2
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Container for detection results."""
    image_path: str
    face_detections: List[dict]
    plate_detections: List[dict]
    image_width: int
    image_height: int
    processing_time: float


class OptimizedDualModelDetector:
    """Optimized dual-model detector with multi-threading and batching."""
    
    def __init__(
        self,
        face_model_path: str,
        plate_model_path: str,
        devices: List[str] = ['cuda:0'],
        slice_height: int = 1280,
        slice_width: int = 1280,
        overlap_ratio: float = 0.2,
        num_workers: int = 4
    ):
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_ratio = overlap_ratio
        self.num_workers = num_workers
        self.devices = devices
        self.num_gpus = len(devices)
        
        # Load models on each GPU
        self.face_models = []
        self.plate_models = []
        
        logger.info(f"Initializing models on {self.num_gpus} GPU(s): {devices}")
        
        for device in devices:
            logger.info(f"Loading face model on {device}...")
            face_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path=face_model_path,
                device=device
            )
            self.face_models.append(face_model)
            
            logger.info(f"Loading plate model on {device}...")
            plate_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path=plate_model_path,
                device=device
            )
            self.plate_models.append(plate_model)
        
        logger.info("✓ All models loaded successfully")
        
        # Model assignment counter for load balancing
        self.model_counter = 0
        self.counter_lock = threading.Lock()
    
    def get_next_models(self) -> Tuple[AutoDetectionModel, AutoDetectionModel, str]:
        """Get next available model pair in round-robin fashion."""
        with self.counter_lock:
            idx = self.model_counter % self.num_gpus
            self.model_counter += 1
        
        return self.face_models[idx], self.plate_models[idx], self.devices[idx]
    
    def detect_image(self, image_path: str) -> Tuple[object, object]:
        """
        Run both models on an image using next available GPU.
        
        Returns:
            Tuple of (face_result, plate_result)
        """
        import time
        start_time = time.time()
        
        face_model, plate_model, device = self.get_next_models()
        
        # Run both models (can be done in parallel if memory allows)
        # For now, sequential but on optimally assigned GPU
        face_result = get_sliced_prediction(
            image_path,
            face_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            perform_standard_pred=False
        )
        
        plate_result = get_sliced_prediction(
            image_path,
            plate_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            perform_standard_pred=False
        )
        
        processing_time = time.time() - start_time
        
        return face_result, plate_result, processing_time


def convert_to_yolo_format(bbox, image_width: int, image_height: int, class_id: int, confidence: float) -> dict:
    """Convert bounding box to YOLO format."""
    minx, miny, maxx, maxy = bbox
    
    x_center = (minx + maxx) / 2.0 / image_width
    y_center = (miny + maxy) / 2.0 / image_height
    width = (maxx - minx) / image_width
    height = (maxy - miny) / image_height
    
    return {
        'class_id': class_id,
        'x_center': x_center,
        'y_center': y_center,
        'width': width,
        'height': height,
        'confidence': confidence
    }


def save_visualization(
    image_path: Path,
    face_result,
    plate_result,
    output_dir: Path
):
    """
    Save visualization with both face and plate detections on the same image.
    
    Args:
        image_path: Path to the original image
        face_result: SAHI result object for faces
        plate_result: SAHI result object for plates
        output_dir: Directory to save visualization
    """
    try:
        # Ensure output_dir is a Path object
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Could not read image for visualization: {image_path}")
            return
        
        # Make a copy for drawing
        vis_image = image.copy()
        
        # Define colors (BGR format)
        FACE_COLOR = (255, 0, 0)  # Blue for faces
        PLATE_COLOR = (0, 255, 0)  # Green for plates
        
        # Draw face detections
        for obj in face_result.object_prediction_list:
            bbox = obj.bbox.to_voc_bbox()  # [minx, miny, maxx, maxy]
            conf = obj.score.value
            
            # Draw rectangle
            cv2.rectangle(
                vis_image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                FACE_COLOR,
                3
            )
            
            # Draw label
            label = f"Face: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                vis_image,
                (int(bbox[0]), int(bbox[1]) - label_size[1] - 10),
                (int(bbox[0]) + label_size[0], int(bbox[1])),
                FACE_COLOR,
                -1
            )
            cv2.putText(
                vis_image,
                label,
                (int(bbox[0]), int(bbox[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Draw plate detections
        for obj in plate_result.object_prediction_list:
            bbox = obj.bbox.to_voc_bbox()  # [minx, miny, maxx, maxy]
            conf = obj.score.value
            
            # Draw rectangle
            cv2.rectangle(
                vis_image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                PLATE_COLOR,
                3
            )
            
            # Draw label
            label = f"Plate: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                vis_image,
                (int(bbox[0]), int(bbox[1]) - label_size[1] - 10),
                (int(bbox[0]) + label_size[0], int(bbox[1])),
                PLATE_COLOR,
                -1
            )
            cv2.putText(
                vis_image,
                label,
                (int(bbox[0]), int(bbox[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Add legend
        legend_height = 60
        legend = np.zeros((legend_height, vis_image.shape[1], 3), dtype=np.uint8)
        cv2.putText(legend, "Blue = Faces", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, FACE_COLOR, 2)
        cv2.putText(legend, "Green = Plates", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, PLATE_COLOR, 2)
        
        # Combine image and legend
        final_image = np.vstack([vis_image, legend])
        
        # Save visualization
        output_path = output_dir / f"{image_path.stem}_detections.jpg"
        cv2.imwrite(str(output_path), final_image)
        
        logger.info(f"  ✓ Saved visualization to {output_path}")
        
    except Exception as e:
        logger.warning(f"  Failed to save visualization for {image_path.name}: {e}")


def process_single_image(
    image_path: Path,
    detector: OptimizedDualModelDetector,
    visualize: bool,
    visualization_dir: Path = None
) -> DetectionResult:
    """Process a single image with both models."""
    try:
        # Run detection
        face_result, plate_result, proc_time = detector.detect_image(str(image_path))
        
        # Get image dimensions
        image_width = face_result.image_width
        image_height = face_result.image_height
        
        # Convert face detections (class_id = 0)
        face_detections = []
        for obj in face_result.object_prediction_list:
            bbox = obj.bbox.to_xyxy()
            confidence = obj.score.value
            det = convert_to_yolo_format(
                bbox, image_width, image_height,
                class_id=0, confidence=confidence
            )
            face_detections.append(det)
        
        # Convert plate detections (class_id = 1)
        plate_detections = []
        for obj in plate_result.object_prediction_list:
            bbox = obj.bbox.to_xyxy()
            confidence = obj.score.value
            det = convert_to_yolo_format(
                bbox, image_width, image_height,
                class_id=1, confidence=confidence
            )
            plate_detections.append(det)
        
        # Save visualization if requested
        if visualize and visualization_dir:
            save_visualization(
                image_path,
                face_result,
                plate_result,
                visualization_dir  # Can be str or Path, function handles both
            )
        
        return DetectionResult(
            image_path=str(image_path),
            face_detections=face_detections,
            plate_detections=plate_detections,
            image_width=image_width,
            image_height=image_height,
            processing_time=proc_time
        )
    
    except Exception as e:
        logger.error(f"Failed to process {image_path}: {e}")
        return None


class AsyncFileWriter:
    """Asynchronous file writer with buffering."""
    
    def __init__(self, output_file: str, buffer_size: int = 100):
        self.output_file = output_file
        self.buffer_size = buffer_size
        self.buffer = []
        self.lock = threading.Lock()
        self.file_handle = None
        
    def __enter__(self):
        self.file_handle = open(self.output_file, 'a')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        if self.file_handle:
            self.file_handle.close()
    
    def write_detection(self, image_path: str, detection: dict):
        """Buffer a detection for writing."""
        line = (
            f"{image_path} {detection['class_id']} "
            f"{detection['x_center']:.6f} {detection['y_center']:.6f} "
            f"{detection['width']:.6f} {detection['height']:.6f} "
            f"{detection['confidence']:.6f}\n"
        )
        
        with self.lock:
            self.buffer.append(line)
            if len(self.buffer) >= self.buffer_size:
                self._flush_unsafe()
    
    def _flush_unsafe(self):
        """Flush buffer without acquiring lock (caller must hold lock)."""
        if self.buffer and self.file_handle:
            self.file_handle.writelines(self.buffer)
            self.file_handle.flush()
            self.buffer.clear()
    
    def flush(self):
        """Flush buffer to disk."""
        with self.lock:
            self._flush_unsafe()


def load_processed_images(progress_file: str) -> Set[str]:
    """Load set of already processed images from progress file."""
    if not os.path.exists(progress_file):
        return set()

    processed = set()
    with open(progress_file, 'r') as f:
        for line in f:
            if line.strip():
                # Normalize paths to ensure consistency
                processed.add(str(Path(line.strip()).resolve()))

    logger.info(f"Loaded {len(processed)} already processed images")
    return processed


def save_progress(progress_file: str, image_path: str):
    """Thread-safe progress saving."""
    # Use file lock for thread safety
    progress_file = str(progress_file)  # Ensure it's a string, not a Path object
    try:
        try:
            import fcntl
            with open(progress_file, 'a') as f:
                # This works on Unix-like systems; Windows needs different approach
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except (AttributeError, ImportError):
                    pass  # Skip locking on Windows
                # Normalize path before saving
                f.write(f"{str(Path(image_path).resolve())}\n")
                f.flush()
        except ImportError:
            # fcntl not available on Windows, just write without locking
            with open(progress_file, 'a') as f:
                f.write(f"{str(Path(image_path).resolve())}\n")
                f.flush()
    except Exception as e:
        logger.warning(f"Failed to save progress: {e}")


def find_images(
    input_dir: str,
    dir_filter: str = None,
    image_extensions: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
) -> List[Path]:
    """Find all images in input directory, optionally filtering by directory name."""
    input_path = Path(input_dir)
    image_files = []
    
    for ext in image_extensions:
        for img_path in input_path.rglob(f'*{ext}'):
            if dir_filter:
                if any(dir_filter.lower() in part.lower() for part in img_path.parts):
                    image_files.append(img_path)
            else:
                image_files.append(img_path)
        
        for img_path in input_path.rglob(f'*{ext.upper()}'):
            if dir_filter:
                if any(dir_filter.lower() in part.lower() for part in img_path.parts):
                    image_files.append(img_path)
            else:
                image_files.append(img_path)
    
    return list(set(image_files))


def process_images_optimized(
    input_dir: str,
    output_file: str,
    face_model_path: str,
    plate_model_path: str,
    devices: List[str] = ['cuda:0'],
    slice_height: int = 1280,
    slice_width: int = 1280,
    overlap_ratio: float = 0.2,
    visualize_samples: int = 0,
    visualization_dir: str = None,
    resume: bool = True,
    dir_filter: str = None,
    num_workers: int = None,
    batch_size: int = 1
):
    """
    Optimized multi-threaded processing with GPU load balancing.
    
    Args:
        input_dir: Root directory containing images
        output_file: Path to master detection file
        face_model_path: Path to face detection model
        plate_model_path: Path to plate detection model
        devices: List of GPU devices to use (e.g., ['cuda:0', 'cuda:1'])
        slice_height: Height of image slices for SAHI
        slice_width: Width of image slices for SAHI
        overlap_ratio: Overlap ratio for SAHI
        visualize_samples: Number of random samples to visualize
        visualization_dir: Directory to save visualizations
        resume: Whether to resume from previous run
        dir_filter: Only process directories containing this text
        num_workers: Number of worker threads (default: 2 * num_gpus)
        batch_size: Batch size for processing (currently not used, placeholder)
    """
    
    # Setup paths
    input_path = Path(input_dir)
    output_path = Path(output_file)
    progress_file = output_path.parent / f"{output_path.stem}_progress.txt"
    
    # Validate input directory
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = len(devices) * 2  # 2 workers per GPU
    
    logger.info(f"Using {len(devices)} GPU(s) with {num_workers} worker threads")
    
    # Load already processed images if resuming
    processed_images = load_processed_images(progress_file) if resume else set()
    
    # Find all images
    logger.info("Searching for images...")
    if dir_filter:
        logger.info(f"Filtering directories containing: '{dir_filter}'")
    
    image_files = find_images(input_dir, dir_filter)
    
    if not image_files:
        logger.warning(f"No images found in {input_dir}")
        return
    
    # Filter out already processed images
    remaining_images = [img for img in image_files if str(img) not in processed_images]
    
    logger.info(f"Found {len(image_files)} total images")
    logger.info(f"Already processed: {len(processed_images)}")
    logger.info(f"Remaining to process: {len(remaining_images)}")
    
    if not remaining_images:
        logger.info("All images already processed!")
        return
    
    # Select random samples for visualization
    visualize_images = set()
    if visualize_samples > 0:
        n_samples = min(visualize_samples, len(remaining_images))
        visualize_images = set(random.sample(remaining_images, n_samples))
        logger.info(f"Selected {n_samples} random images for visualization")
        
        if visualization_dir:
            vis_path = Path(visualization_dir)
            vis_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    try:
        detector = OptimizedDualModelDetector(
            face_model_path=face_model_path,
            plate_model_path=plate_model_path,
            devices=devices,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_ratio=overlap_ratio,
            num_workers=num_workers
        )
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return
    
    # Initialize progress tracking
    total_images = len(remaining_images)
    processed_count = 0
    total_faces = 0
    total_plates = 0
    total_time = 0.0
    
    # Thread-safe counter
    counter_lock = threading.Lock()
    
    # Write header if new file
    mode = 'a' if (resume and output_path.exists()) else 'w'
    if mode == 'w':
        with open(output_path, 'w') as f:
            f.write("# image_path class_id x_center y_center width height confidence\n")
    
    # Process images with thread pool
    with AsyncFileWriter(output_path) as writer:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_image = {}
            for image_path in remaining_images:
                visualize = image_path in visualize_images
                
                future = executor.submit(
                    process_single_image,
                    image_path,
                    detector,
                    visualize,
                    visualization_dir  # Pass as-is, function handles conversion
                )
                future_to_image[future] = image_path
            
            # Process completed tasks
            from concurrent.futures import as_completed
            for future in as_completed(future_to_image):
                image_path = future_to_image[future]
                
                try:
                    result = future.result()
                    
                    if result is not None:
                        # Write detections
                        for det in result.face_detections:
                            writer.write_detection(result.image_path, det)
                        for det in result.plate_detections:
                            writer.write_detection(result.image_path, det)
                        
                        # Update counters
                        with counter_lock:
                            processed_count += 1
                            total_faces += len(result.face_detections)
                            total_plates += len(result.plate_detections)
                            total_time += result.processing_time
                            
                            # Log progress
                            avg_time = total_time / processed_count
                            remaining = total_images - processed_count
                            eta_seconds = remaining * avg_time
                            eta_minutes = eta_seconds / 60
                            
                            logger.info(
                                f"[{processed_count}/{total_images}] {image_path.name} - "
                                f"{len(result.face_detections)} faces, {len(result.plate_detections)} plates - "
                                f"{result.processing_time:.2f}s - "
                                f"ETA: {eta_minutes:.1f}m"
                            )
                        
                        # Save progress
                        save_progress(str(progress_file), str(image_path))
                    
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
    
    # Final statistics
    logger.info("=" * 80)
    logger.info("Processing complete!")
    logger.info(f"Total images processed: {processed_count}")
    logger.info(f"Total faces detected: {total_faces}")
    logger.info(f"Total plates detected: {total_plates}")
    logger.info(f"Average time per image: {total_time/processed_count:.2f}s")
    logger.info(f"Total processing time: {total_time/60:.1f} minutes")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Progress tracking: {progress_file}")
    if visualize_samples > 0:
        logger.info(f"Visualizations saved to: {visualization_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Optimized dual-model SAHI detection with multi-GPU support'
    )
    
    # Config file argument
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON config file'
    )
    
    # Required arguments (optional if config file is provided)
    parser.add_argument('--input-dir', type=str, default=None)
    parser.add_argument('--output-file', type=str, default=None)
    parser.add_argument('--face-model', type=str, default=None)
    parser.add_argument('--plate-model', type=str, default=None)
    
    # Optional arguments
    parser.add_argument('--devices', type=str, nargs='+', default=None,
                        help='GPU devices to use (e.g., cuda:0 cuda:1)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Single device (overridden by --devices)')
    parser.add_argument('--slice-height', type=int, default=1280)
    parser.add_argument('--slice-width', type=int, default=1280)
    parser.add_argument('--overlap-ratio', type=float, default=0.2)
    parser.add_argument('--visualize', type=int, default=0)
    parser.add_argument('--visualization-dir', type=str, default=None)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--dir-filter', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of worker threads (default: 2 * num_gpus)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for processing (future optimization)')
    
    args = parser.parse_args()
    
    # Load config file if provided
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from: {args.config}")
            if 'description' in config:
                logger.info(f"Config: {config['description']}")
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            return
    
    # Merge config file with command line arguments
    def get_value(cli_arg, config_key, required=False):
        value = cli_arg if cli_arg is not None else config.get(config_key)
        if required and value is None:
            parser.error(f"--{config_key.replace('_', '-')} is required")
        return value
    
    # Get all parameters
    input_dir = get_value(args.input_dir, 'input_dir', required=True)
    output_file = get_value(args.output_file, 'output_file', required=True)
    face_model = get_value(args.face_model, 'face_model', required=True)
    plate_model = get_value(args.plate_model, 'plate_model', required=True)
    slice_height = get_value(args.slice_height, 'slice_height')
    slice_width = get_value(args.slice_width, 'slice_width')
    overlap_ratio = get_value(args.overlap_ratio, 'overlap_ratio')
    visualize = get_value(args.visualize, 'visualize')
    visualization_dir = get_value(args.visualization_dir, 'visualization_dir')
    dir_filter = get_value(args.dir_filter, 'dir_filter')
    num_workers = get_value(args.num_workers, 'num_workers')
    batch_size = get_value(args.batch_size, 'batch_size')
    
    # Handle devices (multi-GPU support)
    if args.devices:
        devices = args.devices
    elif 'devices' in config:
        devices = config['devices']
    else:
        device = get_value(args.device, 'device')
        devices = [device]
    
    # Validate devices
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {available_gpus}")
        for dev in devices:
            if dev.startswith('cuda:'):
                gpu_id = int(dev.split(':')[1])
                if gpu_id >= available_gpus:
                    logger.warning(f"GPU {dev} not available, only {available_gpus} GPUs detected")
    else:
        logger.warning("CUDA not available, falling back to CPU")
        devices = ['cpu']
    
    # Handle resume flag
    if args.no_resume:
        resume = False
    else:
        resume = config.get('resume', True)
    
    # Validate visualization arguments
    if visualize > 0 and not visualization_dir:
        parser.error("--visualization-dir is required when --visualize > 0")
    
    # Run optimized processing
    process_images_optimized(
        input_dir=input_dir,
        output_file=output_file,
        face_model_path=face_model,
        plate_model_path=plate_model,
        devices=devices,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_ratio=overlap_ratio,
        visualize_samples=visualize,
        visualization_dir=visualization_dir,
        resume=resume,
        dir_filter=dir_filter,
        num_workers=num_workers,
        batch_size=batch_size
    )


if __name__ == "__main__":
    main()
