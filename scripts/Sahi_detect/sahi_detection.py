import os
import json
import csv
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


class SAHIDetectionConfig:
    """Load and manage configuration from YAML file."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict:
        """Load YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")
    
    def _validate_config(self) -> None:
        """Validate essential configuration parameters."""
        required_sections = ['directories', 'model', 'slicing', 'image_processing', 'output', 'logging']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate model path exists
        model_path = self.config['model']['path']
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def get(self, section: str, key: str, default=None):
        """Get configuration value with dot notation support."""
        if section not in self.config:
            return default
        return self.config[section].get(key, default)
    
    def __getitem__(self, section: str) -> Dict:
        """Allow dictionary-style access to configuration sections."""
        return self.config.get(section, {})


class SAHIDetectionLogger:
    """Setup and manage logging for the detection pipeline."""
    
    def __init__(self, config: SAHIDetectionConfig):
        """
        Initialize logger based on configuration.
        
        Args:
            config: SAHIDetectionConfig instance
        """
        self.log_config = config['logging']
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with file and console handlers."""
        logger = logging.getLogger('SAHIDetection')
        logger.setLevel(self.log_config['level'])
        
        # Create formatters
        formatter = logging.Formatter(self.log_config['format'])
        
        # Console handler with UTF-8 encoding for Windows compatibility
        if self.log_config['console_output']:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            # Handle Unicode characters on Windows
            if hasattr(console_handler, 'stream'):
                try:
                    console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
                except (AttributeError, ValueError):
                    pass
            logger.addHandler(console_handler)
        
        # File handler
        if self.log_config['log_to_file']:
            log_dir = Path(self.log_config['log_file']).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(
                self.log_config['log_file'],
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def get_logger(self) -> logging.Logger:
        """Return the configured logger instance."""
        return self.logger


class SAHIDetectionPipeline:
    """Main detection pipeline using SAHI with sliced inference."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the detection pipeline.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = SAHIDetectionConfig(config_path)
        log_handler = SAHIDetectionLogger(self.config)
        self.logger = log_handler.get_logger()
        self.detection_model = None
        self.results_summary = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'detections': []
        }
    
    def setup(self) -> bool:
        """
        Setup pipeline: create directories and load model.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            self._create_output_directories()
            self._load_model()
            return True
        except Exception as e:
            self.logger.error(f"Pipeline setup failed: {e}")
            return False
    
    def _create_output_directories(self) -> None:
        """Create output and logging directories."""
        output_dir = Path(self.config.get('directories', 'output_dir'))
        logs_dir = Path(self.config.get('directories', 'logs_dir'))
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Output directory ready: {output_dir}")
    
    def _load_model(self) -> None:
        """Load the detection model."""
        model_config = self.config['model']
        
        self.logger.info(f"Loading model from {model_config['path']}...")
        
        try:
            self.detection_model = AutoDetectionModel.from_pretrained(
                model_type=model_config['type'],
                model_path=model_config['path'],
                device=model_config['device'],
                confidence_threshold=model_config.get('confidence_threshold', 0.25)
            )
            self.logger.info("[OK] Model loaded successfully")
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to load model: {e}")
            raise
    
    def find_images(self) -> List[Path]:
        """
        Find all images in input directory.
        
        Returns:
            List of Path objects for images found
        """
        input_dir = Path(self.config.get('directories', 'input_dir'))
        image_config = self.config['image_processing']
        
        if not input_dir.exists():
            self.logger.error(f"Input directory does not exist: {input_dir}")
            return []
        
        image_files = []
        supported_formats = image_config.get('supported_formats', ['.jpg', '.png'])
        
        # Search for images
        if image_config.get('recursive_search', True):
            for ext in supported_formats:
                image_files.extend(input_dir.rglob(f'*{ext}'))
                
                # Case insensitive search
                if image_config.get('case_insensitive', True):
                    image_files.extend(input_dir.rglob(f'*{ext.upper()}'))
        else:
            # Non-recursive search
            for ext in supported_formats:
                image_files.extend(input_dir.glob(f'*{ext}'))
                
                if image_config.get('case_insensitive', True):
                    image_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        # Remove duplicates
        image_files = list(set(image_files))
        image_files.sort()
        
        # Apply max_images limit
        max_images = image_config.get('max_images')
        if max_images:
            image_files = image_files[:max_images]
        
        self.logger.info(f"Found {len(image_files)} images to process")
        return image_files
    
    def process_images(self, image_files: List[Path]) -> None:
        """
        Process images with detection.
        
        Args:
            image_files: List of image file paths to process
        """
        input_dir = Path(self.config.get('directories', 'input_dir'))
        output_dir = Path(self.config.get('directories', 'output_dir'))
        
        self.results_summary['total_images'] = len(image_files)
        
        for idx, image_path in enumerate(image_files, 1):
            try:
                self._process_single_image(image_path, input_dir, output_dir, idx, len(image_files))
            except Exception as e:
                self.logger.error(f"[ERROR] Failed to process {image_path}: {e}")
                self.results_summary['failed'] += 1
                continue
        
        self._log_summary()
    
    def _process_single_image(self, image_path: Path, input_dir: Path, output_dir: Path, 
                              current: int, total: int) -> None:
        """
        Process a single image.
        
        Args:
            image_path: Path to the image file
            input_dir: Input directory root
            output_dir: Output directory root
            current: Current image number
            total: Total images to process
        """
        # Calculate relative path
        relative_path = image_path.relative_to(input_dir)
        
        # Create output directory structure
        if self.config.get('output', 'preserve_folder_structure'):
            output_image_dir = output_dir / relative_path.parent
        else:
            output_image_dir = output_dir
        
        output_image_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"[{current}/{total}] Processing: {relative_path}")
        
        # Run detection with slicing
        result = self._run_sliced_detection(image_path)
        
        # Export results
        self._export_results(result, output_image_dir, image_path, relative_path)
        
        self.results_summary['successful'] += 1
    
    def _run_sliced_detection(self, image_path: Path):
        """
        Run sliced detection on image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Detection result object
        """
        slicing_config = self.config['slicing']
        
        result = get_sliced_prediction(
            str(image_path),
            self.detection_model,
            slice_height=slicing_config.get('slice_height', 1280),
            slice_width=slicing_config.get('slice_width', 1280),
            overlap_height_ratio=slicing_config.get('overlap_height_ratio', 0.2),
            overlap_width_ratio=slicing_config.get('overlap_width_ratio', 0.2),
            perform_standard_pred=slicing_config.get('perform_standard_pred', False)
        )
        
        return result
    
    def _export_results(self, result, output_image_dir: Path, image_path: Path, 
                        relative_path: Path) -> None:
        """
        Export detection results in configured formats.
        
        Args:
            result: Detection result object
            output_image_dir: Directory to save results
            image_path: Original image path
            relative_path: Relative path for logging
        """
        output_config = self.config['output']
        
        # Export visualized results (annotated images)
        if output_config.get('export_visuals', True):
            # SAHI's export_visuals only accepts export_dir and file_name
            result.export_visuals(
                export_dir=str(output_image_dir),
                file_name=image_path.stem
            )
            self.logger.info(f"[OK] Saved visuals to: {output_image_dir}")
            
            # Convert to JPG if configured
            visual_format = output_config.get('visual_format', 'png').lower()
            if visual_format == 'jpg':
                self._convert_png_to_jpg(output_image_dir, image_path)
        
        # Export as JSON
        if output_config.get('export_results_json', False):
            self._export_json(result, output_image_dir, image_path)
        
        # Export as CSV
        if output_config.get('export_results_csv', False):
            self._export_csv(result, output_image_dir, image_path)
        
        # Export as YOLO format
        if output_config.get('export_results_yolo', False):
            self._export_yolo(result, output_image_dir, image_path)
        
        # Store detection summary
        self.results_summary['detections'].append({
            'image': str(relative_path),
            'detections': len(result.object_prediction_list),
            'timestamp': datetime.now().isoformat()
        })
    
    def _export_json(self, result, output_dir: Path, image_path: Path) -> None:
        """Export results as JSON."""
        try:
            json_path = output_dir / f"{image_path.stem}_results.json"
            
            detections = []
            for pred in result.object_prediction_list:
                detections.append({
                    'class': pred.category.name,
                    'confidence': float(pred.score.value),
                    'bbox': {
                        'x_min': pred.bbox.minx,
                        'y_min': pred.bbox.miny,
                        'x_max': pred.bbox.maxx,
                        'y_max': pred.bbox.maxy
                    }
                })
            
            with open(json_path, 'w') as f:
                json.dump(detections, f, indent=2)
            
            self.logger.debug(f"[OK] Saved JSON results to: {json_path}")
        except Exception as e:
            self.logger.warning(f"Failed to export JSON: {e}")
    
    def _export_csv(self, result, output_dir: Path, image_path: Path) -> None:
        """Export results as CSV."""
        try:
            csv_path = output_dir / f"{image_path.stem}_results.csv"
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Class', 'Confidence', 'X_Min', 'Y_Min', 'X_Max', 'Y_Max'])
                
                for pred in result.object_prediction_list:
                    writer.writerow([
                        pred.category.name,
                        pred.score.value,
                        pred.bbox.minx,
                        pred.bbox.miny,
                        pred.bbox.maxx,
                        pred.bbox.maxy
                    ])
            
            self.logger.debug(f"[OK] Saved CSV results to: {csv_path}")
        except Exception as e:
            self.logger.warning(f"Failed to export CSV: {e}")
    
    
    def _convert_png_to_jpg(self, output_dir: Path, image_path: Path) -> None:
        """
        Convert PNG visualization to JPG with quality control.
        
        Args:
            output_dir: Directory containing the exported PNG
            image_path: Original image path (for filename reference)
        """
        try:
            from PIL import Image
            
            output_config = self.config['output']
            jpg_quality = output_config.get('jpg_quality', 95)
            
            # Validate quality range
            if not (1 <= jpg_quality <= 100):
                self.logger.warning(f"Invalid JPG quality {jpg_quality}, using default 95")
                jpg_quality = 95
            
            # SAHI exports PNG as: {stem}_prediction.png
            png_path = output_dir / f"{image_path.stem}_prediction.png"
            jpg_path = output_dir / f"{image_path.stem}_prediction.jpg"
            
            if not png_path.exists():
                self.logger.debug(f"PNG file not found: {png_path}")
                return
            
            # Open and convert PNG to JPG
            img = Image.open(png_path)
            
            # Convert to RGB if necessary (JPG doesn't support transparency)
            if img.mode != 'RGB':
                if img.mode == 'RGBA':
                    # Convert RGBA to RGB with white background
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[3])
                    img = rgb_img
                else:
                    img = img.convert('RGB')
            
            # Save as JPG
            img.save(jpg_path, 'JPEG', quality=jpg_quality, optimize=True)
            
            # Delete PNG to save space
            png_path.unlink()
            
            self.logger.debug(f"[OK] Converted to JPG (quality={jpg_quality}): {jpg_path.name}")
        
        except ImportError:
            self.logger.error("PIL/Pillow not installed. Install with: pip install Pillow")
        except Exception as e:
            self.logger.error(f"Failed to convert PNG to JPG: {e}")
    
    def _export_yolo(self, result, output_dir: Path, image_path: Path) -> None:
        """
        Export results in YOLO format (.txt files).
        
        YOLO format: <class_id> <x_center> <y_center> <width> <height>
        Where coordinates are normalized (0-1) or in pixels based on config.
        """
        try:
            yolo_path = output_dir / f"{image_path.stem}.txt"
            yolo_format = self.config.get('output', 'yolo_format', 'normalized')
            
            # Get image dimensions for normalization
            from PIL import Image
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            with open(yolo_path, 'w') as f:
                for pred in result.object_prediction_list:
                    # Get bounding box coordinates
                    x_min = pred.bbox.minx
                    y_min = pred.bbox.miny
                    x_max = pred.bbox.maxx
                    y_max = pred.bbox.maxy
                    
                    # Calculate center and dimensions
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # Normalize if required
                    if yolo_format == 'normalized':
                        x_center = x_center / img_width
                        y_center = y_center / img_height
                        width = width / img_width
                        height = height / img_height
                    
                    # Ensure normalized values are within [0, 1]
                    if yolo_format == 'normalized':
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))
                    
                    # Get class ID from category name
                    # Try to extract class ID from name (e.g., "0" from "class_0" or just use name)
                    class_name = pred.category.name
                    try:
                        class_id = int(class_name) if class_name.isdigit() else pred.category.id
                    except (ValueError, AttributeError):
                        class_id = 0
                    
                    # Write YOLO format line
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            self.logger.debug(f"[OK] Saved YOLO format results to: {yolo_path}")
        except Exception as e:
            self.logger.warning(f"Failed to export YOLO format: {e}")
    
    def _log_summary(self) -> None:
        """Log processing summary."""
        self.logger.info("=" * 50)
        self.logger.info("Processing Summary:")
        self.logger.info(f"  Total Images: {self.results_summary['total_images']}")
        self.logger.info(f"  Successful: {self.results_summary['successful']}")
        self.logger.info(f"  Failed: {self.results_summary['failed']}")
        self.logger.info("=" * 50)
        
        output_dir = self.config.get('directories', 'output_dir')
        self.logger.info(f"Results saved to: {output_dir}")
    
    def run(self) -> bool:
        """
        Run the complete detection pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Starting SAHI Detection Pipeline...")
        self.logger.info(f"Configuration file: {self.config.config_path}")
        
        if not self.setup():
            return False
        
        image_files = self.find_images()
        if not image_files:
            self.logger.warning("No images found to process")
            return False
        
        self.process_images(image_files)
        return True


def main():
    """Main entry point."""
    config_file = 'config.yaml'  # Change this to your config file path if needed
    
    try:
        pipeline = SAHIDetectionPipeline(config_file)
        success = pipeline.run()
        
        if success:
            print("\n[OK] Detection pipeline completed successfully!")
        else:
            print("\n[ERROR] Detection pipeline encountered errors")
            return 1
    
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
