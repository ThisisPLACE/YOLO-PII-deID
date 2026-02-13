import os
import argparse
from pathlib import Path
import logging
import cv2
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_file=None):
    """Load configuration from JSON file."""
    
    default_config = {
        "input_dir": r"D:\PLACE - Zotac\BGD\Testing\Testing Collection Stitched",
        "output_dir": r"D:\PLACE - Zotac\BGD\Testing\visualize_detect",
        "crops_dir": r"D:\PLACE - Zotac\BGD\Testing\crops_detect",
        "save_crops": True,
        "line_thickness": 2,
        "font_scale": 0.6,
        "jpg_quality": 85
    }
    
    config = default_config.copy()
    
    if config_file and Path(config_file).exists():
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
                logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load config file {config_file}: {e}. Using defaults.")
    
    return config

def save_config(config, output_file):
    """Save current configuration to JSON file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")

def parse_detections_from_text(text_file_path):
    """Parse detection text file and return list of detections."""
    detections = []
    
    try:
        with open(text_file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#') or line.startswith('-') or line.startswith('Total'):
                continue
            
            # Parse detection line
            try:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 9:
                    class_id = int(parts[0])
                    class_name = parts[1]
                    confidence = float(parts[2])
                    x_min = int(float(parts[3]))
                    y_min = int(float(parts[4]))
                    x_max = int(float(parts[5]))
                    y_max = int(float(parts[6]))
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'x_min': x_min,
                        'y_min': y_min,
                        'x_max': x_max,
                        'y_max': y_max
                    })
            except (ValueError, IndexError):
                continue
    
    except Exception as e:
        logger.warning(f"Error parsing detections from {text_file_path}: {e}")
    
    return detections

def get_color_for_class(class_id):
    """Get BGR color based on class ID (deterministic)."""
    colors = [
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 128),    # Purple
        (128, 128, 0),    # Teal
        (0, 128, 128),    # Dark cyan
        (128, 0, 0),      # Dark red
    ]
    return colors[class_id % len(colors)]

def draw_detections_on_image(image_path, detections, output_path, line_thickness=2, font_scale=0.6):
    """Draw detection boxes on image and save."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return False
        
        for detection in detections:
            x_min = detection['x_min']
            y_min = detection['y_min']
            x_max = detection['x_max']
            y_max = detection['y_max']
            class_name = detection['class_name']
            confidence = detection['confidence']
            class_id = detection['class_id']
            
            # Get color for this class
            color = get_color_for_class(class_id)
            
            # Draw bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, line_thickness)
            
            # Create label with class name and confidence
            label = f"{class_name} {confidence:.2f}"
            
            # Get text size to draw background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                1
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                image,
                (x_min, y_min - text_height - baseline - 5),
                (x_min + text_width, y_min),
                color,
                -1
            )
            
            # Put text on image
            cv2.putText(
                image,
                label,
                (x_min, y_min - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1
            )
        
        # Save image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        return True
    
    except Exception as e:
        logger.error(f"Error drawing detections: {e}")
        return False

def save_detection_crops(image_path, detections, crops_dir, jpg_quality=85):
    """Save cropped regions for each detection."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return 0
        
        crops_saved = 0
        
        for idx, detection in enumerate(detections):
            x_min = detection['x_min']
            y_min = detection['y_min']
            x_max = detection['x_max']
            y_max = detection['y_max']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Extract crop
            crop = image[y_min:y_max, x_min:x_max]
            
            if crop.size == 0:
                continue
            
            # Create crop filename
            crop_filename = f"{image_path.stem}_{class_name.replace(' ', '_')}_{idx}_{confidence:.2f}.jpg"
            crop_path = crops_dir / crop_filename
            
            # Save crop
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
            crops_saved += 1
        
        return crops_saved
    
    except Exception as e:
        logger.warning(f"Error saving crops: {e}")
        return 0

def process_images_recursive(
    input_dir,
    output_dir,
    crops_dir=None,
    save_crops=True,
    line_thickness=2,
    font_scale=0.6,
    jpg_quality=85
):
    """
    Process all images with detections and create visualizations + crops.
    
    Args:
        input_dir: Directory containing images and *_detect.txt files
        output_dir: Output directory for visualization images
        crops_dir: Output directory for cropped regions
        save_crops: Whether to save cropped regions
        line_thickness: Thickness of bounding box lines
        font_scale: Font scale for labels
        jpg_quality: JPG quality for output (1-100)
    """
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if crops_dir:
        crops_path = Path(crops_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    if save_crops and crops_dir:
        crops_path.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = set()
    for ext in image_extensions:
        image_files.update(input_path.rglob(f'*{ext}'))
        image_files.update(input_path.rglob(f'*{ext.upper()}'))
    
    # Convert set to sorted list
    image_files = sorted(list(image_files))
    
    if not image_files:
        logger.warning(f"No images found in {input_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    logger.info(f"Starting visualization and cropping...")
    
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    total_crops = 0
    
    for idx, image_path in enumerate(image_files, 1):
        try:
            relative_path = image_path.relative_to(input_path)
            
            # Look for Stage 1 detection file
            detection_text_path = image_path.parent / f"{image_path.stem}_detect.txt"
            
            if not detection_text_path.exists():
                logger.debug(f"[{idx}/{len(image_files)}] No detection file for: {relative_path}")
                skipped_count += 1
                continue
            
            # Parse detections
            detections = parse_detections_from_text(detection_text_path)
            
            if not detections:
                logger.debug(f"[{idx}/{len(image_files)}] No detections in: {relative_path}")
                skipped_count += 1
                continue
            
            remaining = len(image_files) - idx
            logger.info(f"[{idx}/{len(image_files)} | Remaining: {remaining}] {relative_path} ({len(detections)} detections)")
            
            # Create output visualization path
            output_subdir = output_path / relative_path.parent
            output_viz_path = output_subdir / f"{image_path.stem}_visualization.jpg"
            
            # Draw and save visualization
            if draw_detections_on_image(image_path, detections, output_viz_path, line_thickness, font_scale):
                # Save with specified quality
                image = cv2.imread(str(output_viz_path))
                cv2.imwrite(str(output_viz_path), image, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
                processed_count += 1
                
                # Save crops if enabled
                if save_crops and crops_dir:
                    crops_count = save_detection_crops(image_path, detections, crops_path, jpg_quality)
                    total_crops += crops_count
            else:
                failed_count += 1
        
        except Exception as e:
            logger.error(f"✗ Failed to process {image_path}: {e}")
            failed_count += 1
            continue
    
    # Summary
    logger.info("=" * 70)
    logger.info("Processing complete!")
    logger.info(f"Processed: {processed_count}")
    logger.info(f"Skipped (no detections): {skipped_count}")
    if failed_count > 0:
        logger.warning(f"Failed: {failed_count}")
    logger.info(f"Output saved to: {output_dir}")
    if save_crops and crops_dir:
        logger.info(f"Crops saved to: {crops_dir}")
        logger.info(f"Total crops saved: {total_crops}")
    logger.info("=" * 70)

def main():
    """Main function with CLI argument support."""
    parser = argparse.ArgumentParser(
        description='Visualize Stage 1 detections from _detect.txt files with optional crop output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize only (no crops)
  python visualize_detections_with_crops.py --no-crops
  
  # Visualize and save crops
  python visualize_detections_with_crops.py --config config_viz.json
  
  # Custom crop directory
  python visualize_detections_with_crops.py --crops-dir /path/to/crops
  
  # Thicker lines and high quality
  python visualize_detections_with_crops.py --line-thickness 3 --jpg-quality 95
        """
    )
    
    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--input-dir', type=str, help='Input directory with images and *_detect.txt files')
    parser.add_argument('--output-dir', type=str, help='Output directory for visualization images')
    parser.add_argument('--crops-dir', type=str, help='Output directory for cropped regions')
    parser.add_argument('--line-thickness', type=int, help='Thickness of bounding box lines (1-10)')
    parser.add_argument('--font-scale', type=float, help='Font scale for labels (0.3-1.5)')
    parser.add_argument('--jpg-quality', type=int, help='JPG quality (1-100)')
    parser.add_argument('--no-crops', action='store_true', help='Do not save cropped regions')
    parser.add_argument('--save-config', type=str, help='Save current config to JSON file')
    parser.add_argument('--show-config', action='store_true', help='Display configuration and exit')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with CLI arguments
    if args.input_dir:
        config['input_dir'] = args.input_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.crops_dir:
        config['crops_dir'] = args.crops_dir
    if args.line_thickness:
        config['line_thickness'] = args.line_thickness
    if args.font_scale:
        config['font_scale'] = args.font_scale
    if args.jpg_quality:
        config['jpg_quality'] = args.jpg_quality
    if args.no_crops:
        config['save_crops'] = False
    
    # Display configuration if requested
    if args.show_config:
        logger.info("Current configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
        return
    
    # Save configuration if requested
    if args.save_config:
        save_config(config, args.save_config)
    
    # Log final configuration
    logger.info("Starting visualization with configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Run visualization and cropping
    process_images_recursive(**config)

if __name__ == "__main__":
    main()
