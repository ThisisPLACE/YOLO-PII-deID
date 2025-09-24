"""
Image Blurring Script with YOLO Annotations

This script applies blurring to specific regions in images based on YOLO format annotation files.
It preserves EXIF metadata including GPS information in the processed images.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from GPSPhoto import gpsphoto
import piexif


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Apply oval blurring with feathered edges to specific regions in images based on YOLO format annotation files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python script.py -i images/
  
  # With custom blur intensity
  python script.py -i images/ --blur-size 60
  
  # Custom output subdirectory name
  python script.py -i images/ --output-subdir blurred_faces
  
  # Different file extensions
  python script.py -i images/ --image-ext .png --annotation-ext .label
        """
    )
    
    # Required arguments
    parser.add_argument(
        "-i", "--images-dir",
        type=str,
        required=True,
        help="Directory containing source images and annotation files"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="blurred",
        help="Name of output subdirectory (default: 'blurred')"
    )
    
    parser.add_argument(
        "--blur-size",
        type=int,
        default=40,
        help="Size of blur kernel (default: 40). Higher values = more blur"
    )
    
    parser.add_argument(
        "--feather-size",
        type=int,
        default=10,
        help="Size of feathering/edge softening (default: 10). Higher values = softer edges"
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip processing if output file already exists (default: True)"
    )
    
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        default=False,
        help="Force reprocess all files, even if output exists"
    )
    
    parser.add_argument(
        "--image-ext",
        type=str,
        default=".jpg",
        help="Image file extension (default: .jpg)"
    )
    
    parser.add_argument(
        "--annotation-ext", 
        type=str,
        default=".txt",
        help="Annotation file extension (default: .txt)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true", 
        default=False,
        help="Show what would be processed without actually processing"
    )
    
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Check directory exists
    if not Path(args.images_dir).exists():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")
    
    # Validate blur and feather sizes
    if args.blur_size <= 0:
        raise ValueError(f"Blur size must be positive, got: {args.blur_size}")
    
    if args.feather_size < 0:
        raise ValueError(f"Feather size must be non-negative, got: {args.feather_size}")
    
    # Handle conflicting skip options
    if args.no_skip_existing:
        args.skip_existing = False
    
    # Ensure extensions start with dot
    if not args.image_ext.startswith('.'):
        args.image_ext = '.' + args.image_ext
    if not args.annotation_ext.startswith('.'):
        args.annotation_ext = '.' + args.annotation_ext
    
    # Set up output directory path
    args.output_dir = str(Path(args.images_dir) / args.output_subdir)


class ImageBlurProcessor:
    """Processes images by applying oval blurring with feathered edges to regions specified in YOLO annotation files."""
    
    def __init__(self, images_dir: str, output_dir: str, 
                 blur_kernel_size: int = 40, feather_size: int = 10, skip_existing: bool = True,
                 image_ext: str = ".jpg", annotation_ext: str = ".txt", 
                 verbose: bool = False, dry_run: bool = False):
        """
        Initialize the processor with directory paths and settings.
        
        Args:
            images_dir: Directory containing source images and annotations
            output_dir: Directory to save blurred images
            blur_kernel_size: Size of blur kernel (default: 40)
            feather_size: Size of feathering/edge softening (default: 10)
            skip_existing: Skip processing if output file exists (default: True)
            image_ext: Image file extension (default: ".jpg")
            annotation_ext: Annotation file extension (default: ".txt")
            verbose: Enable verbose output (default: False)
            dry_run: Show what would be processed without processing (default: False)
        """
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(images_dir)  # Same as images directory
        self.output_dir = Path(output_dir)
        self.blur_kernel_size = blur_kernel_size
        self.feather_size = feather_size
        self.skip_existing = skip_existing
        self.image_ext = image_ext
        self.annotation_ext = annotation_ext
        self.verbose = verbose
        self.dry_run = dry_run
        self.failed_files = []
        
        # Create/overwrite output directory
        if not self.dry_run:
            if self.output_dir.exists():
                self._log(f"Output directory exists and will be overwritten: {self.output_dir}")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate directories
        self._validate_directories()
    
    def _validate_directories(self) -> None:
        """Validate that required directories exist."""
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
    
    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def _copy_exif_metadata(self, source_path: str, target_path: str) -> bool:
        """
        Copy EXIF metadata from source image to target image.
        
        Args:
            source_path: Path to source image
            target_path: Path to target image
            
        Returns:
            True if successful, False otherwise
        """
        if self.dry_run:
            self._log(f"Would copy EXIF from {source_path} to {target_path}")
            return True
            
        try:
            exif_dict = piexif.load(source_path)
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, target_path)
            self._log(f"Copied EXIF metadata to {target_path}")
            return True
        except Exception as e:
            print(f"Warning: Failed to copy EXIF data for {target_path}: {e}")
            return False
    
    def _parse_yolo_annotation(self, line: str) -> Tuple[int, float, float, float, float]:
        """
        Parse a single line of YOLO annotation.
        
        Args:
            line: Annotation line in YOLO format
            
        Returns:
            Tuple of (class_id, x_center, y_center, width, height)
        """
        parts = line.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Invalid annotation format: {line}")
        
        class_id = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        return class_id, x, y, w, h
    
    def _yolo_to_pixel_coords(self, x: float, y: float, w: float, h: float, 
                             img_height: int, img_width: int) -> Tuple[int, int, int, int]:
        """
        Convert YOLO normalized coordinates to pixel coordinates.
        
        Args:
            x, y, w, h: YOLO normalized coordinates
            img_height, img_width: Image dimensions
            
        Returns:
            Tuple of (left_x, top_y, right_x, bottom_y) in pixels
        """
        center_x = x * img_width
        center_y = y * img_height
        box_width = w * img_width
        box_height = h * img_height
        
        left_x = max(0, int(center_x - box_width / 2))
        top_y = max(0, int(center_y - box_height / 2))
        right_x = min(img_width, int(center_x + box_width / 2))
        bottom_y = min(img_height, int(center_y + box_height / 2))
        
        return left_x, top_y, right_x, bottom_y
    
    def _create_oval_mask(self, width: int, height: int, feather_size: int) -> np.ndarray:
        """
        Create an oval mask with feathered edges.
        
        Args:
            width: Width of the mask
            height: Height of the mask
            feather_size: Size of feathering in pixels
            
        Returns:
            Normalized mask array with values 0-1
        """
        # Create coordinate grids
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height / 2, width / 2
        
        # Create ellipse equation: (x-cx)²/a² + (y-cy)²/b² <= 1
        # Use smaller radius to account for feathering
        radius_x = max(1, width / 2 - feather_size)
        radius_y = max(1, height / 2 - feather_size)
        
        # Calculate normalized distance from center
        ellipse_dist = ((x - center_x) ** 2 / radius_x ** 2 + 
                       (y - center_y) ** 2 / radius_y ** 2)
        
        if feather_size > 0:
            # Create smooth transition using gaussian-like falloff
            # Distance of 1.0 = edge of ellipse, > 1.0 = outside
            feather_factor = feather_size / min(width, height) * 4  # Adjust feather strength
            mask = np.exp(-np.maximum(0, ellipse_dist - 1.0) / feather_factor)
            mask = np.clip(mask, 0, 1)
        else:
            # Hard edge
            mask = (ellipse_dist <= 1.0).astype(np.float32)
        
        return mask
    
    def _blur_region(self, image: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
        """
        Apply oval blur with feathered edges to a specific region of the image.
        
        Args:
            image: Input image
            x, y, w, h: YOLO normalized coordinates
            
        Returns:
            Image with blurred region
        """
        img_height, img_width = image.shape[:2]
        left_x, top_y, right_x, bottom_y = self._yolo_to_pixel_coords(
            x, y, w, h, img_height, img_width
        )
        
        # Extract region
        region_width = right_x - left_x
        region_height = bottom_y - top_y
        
        if region_width <= 0 or region_height <= 0:
            return image
        
        # Extract the region to be blurred
        original_region = image[top_y:bottom_y, left_x:right_x].copy()
        
        # Apply blur to the region
        blurred_region = cv2.blur(original_region, (self.blur_kernel_size, self.blur_kernel_size))
        
        # Create oval mask with feathered edges
        mask = self._create_oval_mask(region_width, region_height, self.feather_size)
        
        # Ensure mask has the right shape for broadcasting
        if len(image.shape) == 3:  # Color image
            mask = mask[:, :, np.newaxis]
        
        # Blend original and blurred regions using the mask
        # mask = 1 means fully blurred, mask = 0 means original
        blended_region = (original_region * (1 - mask) + blurred_region * mask).astype(np.uint8)
        
        # Put the blended region back into the image
        image[top_y:bottom_y, left_x:right_x] = blended_region
        
        return image
    
    def _process_single_image(self, annotation_file: Path) -> bool:
        """
        Process a single image based on its annotation file.
        
        Args:
            annotation_file: Path to annotation file
            
        Returns:
            True if successful, False otherwise
        """
        # Determine corresponding image and output paths
        image_name = annotation_file.stem + self.image_ext
        image_file = self.images_dir / image_name
        output_file = self.output_dir / image_name
        
        self._log(f"Processing: {annotation_file.name} -> {image_name}")
        
        # Skip if output already exists and skip_existing is True
        if self.skip_existing and output_file.exists():
            self._log(f"Skipping {output_file} (already exists)")
            return True
        
        # Check if image file exists
        if not image_file.exists():
            print(f"Warning: Image file not found: {image_file}")
            return False
        
        if self.dry_run:
            self._log(f"Would process: {image_file} -> {output_file}")
            return True
        
        try:
            # Load image
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"Error: Could not load image: {image_file}")
                return False
            
            self._log(f"Loaded image: {image.shape}")
            
            # Process annotations
            regions_processed = 0
            with open(annotation_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    try:
                        _, x, y, w, h = self._parse_yolo_annotation(line)
                        image = self._blur_region(image, x, y, w, h)
                        regions_processed += 1
                        self._log(f"Applied oval blur to region {regions_processed}: x={x:.3f}, y={y:.3f}, w={w:.3f}, h={h:.3f}")
                    except Exception as e:
                        print(f"Warning: Error parsing line {line_num} in {annotation_file}: {e}")
                        continue
            
            self._log(f"Processed {regions_processed} regions")
            
            # Save blurred image
            success = cv2.imwrite(str(output_file), image)
            if not success:
                print(f"Error: Could not save image: {output_file}")
                return False
            
            self._log(f"Saved blurred image: {output_file}")
            
            # Copy EXIF metadata
            self._copy_exif_metadata(str(image_file), str(output_file))
            
            return True
            
        except Exception as e:
            print(f"Error processing {annotation_file}: {e}")
            return False
    
    def get_file_counts(self) -> Tuple[int, int]:
        """
        Get counts of annotation and image files.
        
        Returns:
            Tuple of (annotation_count, image_count)
        """
        annotation_files = list(self.annotations_dir.glob(f'*{self.annotation_ext}'))
        image_files = list(self.images_dir.glob(f'*{self.image_ext}'))
        return len(annotation_files), len(image_files)
    
    def process_all_images(self) -> None:
        """Process all images in the input directory."""
        annotation_files = list(self.annotations_dir.glob(f'*{self.annotation_ext}'))
        
        if not annotation_files:
            print(f"No annotation files (*{self.annotation_ext}) found in {self.annotations_dir}")
            return
        
        print(f"Found {len(annotation_files)} annotation files to process")
        
        if self.dry_run:
            print("DRY RUN MODE - No files will be modified")
        
        if self.verbose:
            print(f"Settings:")
            print(f"  Images/Annotations dir: {self.images_dir}")
            print(f"  Output dir: {self.output_dir}")
            print(f"  Blur kernel size: {self.blur_kernel_size}")
            print(f"  Feather size: {self.feather_size}")
            print(f"  Skip existing: {self.skip_existing}")
            print(f"  Image extension: {self.image_ext}")
            print(f"  Annotation extension: {self.annotation_ext}")
            print()
        
        processed_count = 0
        failed_count = 0
        skipped_count = 0
        
        for i, ann_file in enumerate(annotation_files):
            # Show progress
            if not self.verbose:
                print(f'\rProcessing: {i+1}/{len(annotation_files)}', end='', flush=True)
            else:
                print(f'Processing {i+1}/{len(annotation_files)}: {ann_file.name}')
            
            # Check if would be skipped
            output_name = ann_file.stem + self.image_ext
            output_file = self.output_dir / output_name
            
            if self.skip_existing and output_file.exists() and not self.dry_run:
                skipped_count += 1
                continue
            
            if self._process_single_image(ann_file):
                processed_count += 1
            else:
                failed_count += 1
                self.failed_files.append(str(ann_file))
        
        # Print summary
        if not self.verbose:
            print()  # New line after progress indicator
            
        print(f"\nProcessing {'simulation ' if self.dry_run else ''}complete:")
        print(f"Successfully processed: {processed_count} images")
        if skipped_count > 0:
            print(f"Skipped (already exist): {skipped_count} images")
        print(f"Failed to process: {failed_count} images")
        
        if self.failed_files:
            print(f"\nFailed files:")
            for failed_file in self.failed_files:
                print(f"  - {failed_file}")


def main():
    """Main function to run the image blurring process."""
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Validate arguments
        validate_args(args)
        
        # Display configuration
        print("Image Oval Blurring Script")
        print("=" * 50)
        print(f"Images/Annotations directory: {args.images_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Blur kernel size: {args.blur_size}")
        print(f"Feather size: {args.feather_size}")
        print(f"Skip existing files: {args.skip_existing}")
        print(f"Image extension: {args.image_ext}")
        print(f"Annotation extension: {args.annotation_ext}")
        print(f"Verbose mode: {args.verbose}")
        print(f"Dry run mode: {args.dry_run}")
        print("=" * 50)
        
        # Create processor
        processor = ImageBlurProcessor(
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            blur_kernel_size=args.blur_size,
            feather_size=args.feather_size,
            skip_existing=args.skip_existing,
            image_ext=args.image_ext,
            annotation_ext=args.annotation_ext,
            verbose=args.verbose,
            dry_run=args.dry_run
        )
        
        # Show file counts
        ann_count, img_count = processor.get_file_counts()
        print(f"Found {ann_count} annotation files and {img_count} image files")
        
        if ann_count != img_count:
            print("Warning: Number of annotation and image files don't match")
            if args.verbose:
                # Show which files are missing matches
                ann_files = {f.stem for f in Path(args.images_dir).glob(f'*{args.annotation_ext}')}
                img_files = {f.stem for f in Path(args.images_dir).glob(f'*{args.image_ext}')}
                
                missing_images = ann_files - img_files
                missing_annotations = img_files - ann_files
                
                if missing_images:
                    print("Annotation files without matching images:")
                    for name in sorted(missing_images):
                        print(f"  - {name}{args.annotation_ext}")
                
                if missing_annotations:
                    print("Image files without matching annotations:")
                    for name in sorted(missing_annotations):
                        print(f"  - {name}{args.image_ext}")
        
        print()
        
        # Process all images
        processor.process_all_images()
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()