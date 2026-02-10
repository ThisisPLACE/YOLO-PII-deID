import os
import argparse
import random
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import logging
from typing import Set, List, Tuple
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DualModelDetector:
    """Runs two YOLO models on images and combines results."""
    
    def __init__(
        self,
        face_model_path: str,
        plate_model_path: str,
        device: str = 'cuda:0',
        slice_height: int = 1280,
        slice_width: int = 1280,
        overlap_ratio: float = 0.2
    ):
        self.device = device
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_ratio = overlap_ratio
        
        # Load both models
        logger.info("Loading face detection model...")
        self.face_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=face_model_path,
            device=device
        )
        logger.info("✓ Face model loaded")
        
        logger.info("Loading plate detection model...")
        self.plate_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=plate_model_path,
            device=device
        )
        logger.info("✓ Plate model loaded")
    
    def detect_image(self, image_path: str) -> Tuple[List[dict], List[dict]]:
        """
        Run both models on an image and return detections.
        
        Returns:
            Tuple of (face_detections, plate_detections)
        """
        # Run face detection
        face_result = get_sliced_prediction(
            image_path,
            self.face_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            perform_standard_pred=False
        )
        
        # Run plate detection
        plate_result = get_sliced_prediction(
            image_path,
            self.plate_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            perform_standard_pred=False
        )
        
        return face_result, plate_result


def convert_to_yolo_format(bbox, image_width: int, image_height: int, class_id: int, confidence: float) -> dict:
    """
    Convert bounding box to YOLO format.
    
    Args:
        bbox: Bounding box with minx, miny, maxx, maxy
        image_width: Image width
        image_height: Image height
        class_id: Class ID (0 for face, 1 for plate)
        confidence: Detection confidence
    
    Returns:
        Dict with YOLO format coordinates
    """
    minx, miny, maxx, maxy = bbox
    
    # Calculate center, width, height
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


def load_processed_images(progress_file: str) -> Set[str]:
    """Load set of already processed images from progress file."""
    if not os.path.exists(progress_file):
        return set()
    
    processed = set()
    with open(progress_file, 'r') as f:
        for line in f:
            if line.strip():
                processed.add(line.strip())
    
    logger.info(f"Loaded {len(processed)} already processed images")
    return processed


def save_progress(progress_file: str, image_path: str):
    """Append processed image to progress file."""
    with open(progress_file, 'a') as f:
        f.write(f"{image_path}\n")


def find_images(
    input_dir: str,
    dir_filter: str = None,
    image_extensions: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
) -> List[Path]:
    """
    Find all images in input directory, optionally filtering by directory name.
    
    Args:
        input_dir: Root directory to search
        dir_filter: Only process directories containing this text (case-insensitive)
        image_extensions: Set of valid image extensions
    
    Returns:
        List of Path objects for images
    """
    input_path = Path(input_dir)
    image_files = []
    
    for ext in image_extensions:
        for img_path in input_path.rglob(f'*{ext}'):
            # Apply directory filter if specified
            if dir_filter:
                # Check if any parent directory contains the filter text
                if any(dir_filter.lower() in part.lower() for part in img_path.parts):
                    image_files.append(img_path)
            else:
                image_files.append(img_path)
        
        # Also check uppercase extensions
        for img_path in input_path.rglob(f'*{ext.upper()}'):
            if dir_filter:
                if any(dir_filter.lower() in part.lower() for part in img_path.parts):
                    image_files.append(img_path)
            else:
                image_files.append(img_path)
    
    # Remove duplicates
    image_files = list(set(image_files))
    
    return image_files


def process_images(
    input_dir: str,
    output_file: str,
    face_model_path: str,
    plate_model_path: str,
    device: str = 'cuda:0',
    slice_height: int = 1280,
    slice_width: int = 1280,
    overlap_ratio: float = 0.2,
    visualize_samples: int = 0,
    visualization_dir: str = None,
    resume: bool = True,
    dir_filter: str = None
):
    """
    Process all images with dual model detection.
    
    Args:
        input_dir: Root directory containing images
        output_file: Path to master detection file
        face_model_path: Path to face detection model
        plate_model_path: Path to plate detection model
        device: Device to use for inference
        slice_height: Height of image slices for SAHI
        slice_width: Width of image slices for SAHI
        overlap_ratio: Overlap ratio for SAHI
        visualize_samples: Number of random samples to visualize (0 = no visualization)
        visualization_dir: Directory to save visualizations
        resume: Whether to resume from previous run
        dir_filter: Only process directories containing this text
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
        detector = DualModelDetector(
            face_model_path=face_model_path,
            plate_model_path=plate_model_path,
            device=device,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_ratio=overlap_ratio
        )
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return
    
    # Open output file in append mode
    mode = 'a' if (resume and output_path.exists()) else 'w'
    with open(output_path, mode) as out_f:
        # Write header if new file
        if mode == 'w':
            out_f.write("# image_path class_id x_center y_center width height confidence\n")
        
        # Process each image
        for idx, image_path in enumerate(remaining_images, 1):
            try:
                logger.info(f"[{idx}/{len(remaining_images)}] Processing: {image_path.name}")
                
                # Run both models
                face_result, plate_result = detector.detect_image(str(image_path))
                
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
                
                # Combine all detections
                all_detections = face_detections + plate_detections
                
                # Write to output file
                for det in all_detections:
                    out_f.write(
                        f"{image_path} {det['class_id']} "
                        f"{det['x_center']:.6f} {det['y_center']:.6f} "
                        f"{det['width']:.6f} {det['height']:.6f} "
                        f"{det['confidence']:.6f}\n"
                    )
                
                # Save visualization if this image was selected
                if image_path in visualize_images and visualization_dir:
                    try:
                        vis_output_dir = Path(visualization_dir)
                        
                        # Export face detections
                        face_result.export_visuals(
                            export_dir=str(vis_output_dir),
                            file_name=f"{image_path.stem}_faces"
                        )
                        
                        # Export plate detections
                        plate_result.export_visuals(
                            export_dir=str(vis_output_dir),
                            file_name=f"{image_path.stem}_plates"
                        )
                        
                        logger.info(f"  ✓ Saved visualizations to {vis_output_dir}")
                    except Exception as e:
                        logger.warning(f"  Failed to save visualization: {e}")
                
                # Save progress
                save_progress(progress_file, str(image_path))
                
                logger.info(
                    f"  ✓ Found {len(face_detections)} faces, "
                    f"{len(plate_detections)} plates"
                )
                
            except Exception as e:
                logger.error(f"  ✗ Failed to process {image_path}: {e}")
                continue
    
    logger.info("=" * 80)
    logger.info("Processing complete!")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Progress tracking: {progress_file}")
    if visualize_samples > 0:
        logger.info(f"Visualizations saved to: {visualization_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Dual-model SAHI detection with resume and filtering capabilities'
    )
    
    # Config file argument
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON config file (if provided, other arguments are optional)'
    )
    
    # Required arguments (optional if config file is provided)
    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help='Input directory containing images'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output master detection file (YOLO format)'
    )
    parser.add_argument(
        '--face-model',
        type=str,
        default=None,
        help='Path to face detection model'
    )
    parser.add_argument(
        '--plate-model',
        type=str,
        default=None,
        help='Path to plate detection model'
    )
    
    # Optional arguments
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use (default: cuda:0)'
    )
    parser.add_argument(
        '--slice-height',
        type=int,
        default=1280,
        help='Slice height for SAHI (default: 1280)'
    )
    parser.add_argument(
        '--slice-width',
        type=int,
        default=1280,
        help='Slice width for SAHI (default: 1280)'
    )
    parser.add_argument(
        '--overlap-ratio',
        type=float,
        default=0.2,
        help='Overlap ratio for SAHI (default: 0.2)'
    )
    parser.add_argument(
        '--visualize',
        type=int,
        default=0,
        help='Number of random samples to visualize (default: 0 = no visualization)'
    )
    parser.add_argument(
        '--visualization-dir',
        type=str,
        default=None,
        help='Directory to save visualizations (required if --visualize > 0)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh without resuming from previous progress'
    )
    parser.add_argument(
        '--dir-filter',
        type=str,
        default=None,
        help='Only process directories containing this text (e.g., "stitched")'
    )
    
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
    
    # Merge config file with command line arguments (CLI takes precedence)
    def get_value(cli_arg, config_key, required=False):
        """Get value from CLI arg or config file, CLI takes precedence."""
        value = cli_arg if cli_arg is not None else config.get(config_key)
        if required and value is None:
            parser.error(f"--{config_key.replace('_', '-')} is required (not in config or CLI)")
        return value
    
    # Get all parameters
    input_dir = get_value(args.input_dir, 'input_dir', required=True)
    output_file = get_value(args.output_file, 'output_file', required=True)
    face_model = get_value(args.face_model, 'face_model', required=True)
    plate_model = get_value(args.plate_model, 'plate_model', required=True)
    device = get_value(args.device, 'device')
    slice_height = get_value(args.slice_height, 'slice_height')
    slice_width = get_value(args.slice_width, 'slice_width')
    overlap_ratio = get_value(args.overlap_ratio, 'overlap_ratio')
    visualize = get_value(args.visualize, 'visualize')
    visualization_dir = get_value(args.visualization_dir, 'visualization_dir')
    dir_filter = get_value(args.dir_filter, 'dir_filter')
    
    # Handle resume flag (special case - CLI flag overrides config)
    if args.no_resume:
        resume = False
    else:
        resume = config.get('resume', True)
    
    # Validate visualization arguments
    if visualize > 0 and not visualization_dir:
        parser.error("--visualization-dir is required when --visualize > 0")
    
    # Run processing
    process_images(
        input_dir=input_dir,
        output_file=output_file,
        face_model_path=face_model,
        plate_model_path=plate_model,
        device=device,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_ratio=overlap_ratio,
        visualize_samples=visualize,
        visualization_dir=visualization_dir,
        resume=resume,
        dir_filter=dir_filter
    )


if __name__ == "__main__":
    main()
