#!/usr/bin/env python3
"""
Image Stitched Folder Processor - Optimized for Speed

This script processes images from folders containing "stitched" in their path.
It uses multiprocessing for parallel image compression and optimizations for speed.
- Compresses to half resolution and 50% JPG quality
- Strips EXIF data
- Preserves directory structure
- Uses parallel processing for maximum speed
"""

import os
import sys
from pathlib import Path
from PIL import Image
import time
from datetime import timedelta
from multiprocessing import Pool, cpu_count
from functools import partial
import traceback


def format_time(seconds):
    """Format seconds to human-readable time"""
    return str(timedelta(seconds=int(seconds)))


def print_progress_bar(current, total, elapsed_time, prefix='Progress'):
    """
    Print a progress bar with percentage and ETA based on overall progress rate
    
    Args:
        current (int): Current item number (1-indexed)
        total (int): Total items
        elapsed_time (float): Total elapsed time in seconds
        prefix (str): Prefix for the progress bar
    """
    if total == 0:
        return
    
    percent = current / total
    filled = int(50 * percent)
    bar = '█' * filled + '░' * (50 - filled)
    
    # Calculate ETA based on overall progress rate
    if current > 0 and percent < 1.0:
        # Calculate processing rate (items per second)
        processing_rate = current / elapsed_time
        remaining_items = total - current
        eta_seconds = remaining_items / processing_rate
        eta_str = format_time(eta_seconds)
    else:
        eta_str = "calculating..."
    
    elapsed_str = format_time(elapsed_time)
    percentage = percent * 100
    
    print(f'\r{prefix}: |{bar}| {current}/{total} ({percentage:5.1f}%) | Elapsed: {elapsed_str} | ETA: {eta_str}', end='', flush=True)


def compress_image_optimized(args):
    """
    Optimized image compression function for parallel processing.
    
    Args:
        args (tuple): (input_path, output_path, quality)
        
    Returns:
        tuple: (success: bool, error_msg: str or None)
    """
    input_path, output_path, quality = args
    
    try:
        # Open image and convert immediately
        img = Image.open(input_path)
        
        # Faster approach: convert to RGB first to strip EXIF
        if img.mode != 'RGB':
            if img.mode == 'RGBA':
                # Create white background for RGBA
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode == 'P':
                img = img.convert('RGBA')
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            else:
                img = img.convert('RGB')
        
        # Calculate new dimensions (half resolution)
        new_width = img.width // 2
        new_height = img.height // 2
        
        # Faster resampling - use LANCZOS
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save with optimization - use faster settings
        img_resized.save(output_path, 'JPEG', quality=quality, optimize=False)
        
        return (True, None)
        
    except Exception as e:
        return (False, str(e))


def find_stitched_folders(parent_dir):
    """
    Find all folders that contain images and have "stitched" in their path.
    
    Args:
        parent_dir (str): Parent directory path to search
        
    Returns:
        list: List of paths containing "stitched" with image files
    """
    stitched_folders = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    try:
        for root, dirs, files in os.walk(parent_dir):
            # Check if "stitched" is in the path
            if 'stitched' in root.lower():
                # Check if folder contains image files
                image_files = [f for f in files if Path(f).suffix.lower() in image_extensions]
                if image_files:
                    stitched_folders.append(root)
    except Exception as e:
        print(f"Error while searching directories: {e}")
        return []
    
    return stitched_folders


def process_stitched_images(parent_dir, output_dir, num_workers=None):
    """
    Find and process all stitched images using multiprocessing.
    
    Args:
        parent_dir (str): Parent directory to search
        output_dir (str): Output directory for processed images
        num_workers (int): Number of parallel workers (default: CPU count)
    """
    # Validate inputs
    if not os.path.isdir(parent_dir):
        print(f"Error: Parent directory '{parent_dir}' does not exist")
        sys.exit(1)
    
    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory: {e}")
        sys.exit(1)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one core free
    
    print(f"Starting image processing...")
    print(f"Parent directory: {parent_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Using {num_workers} parallel worker(s)")
    print()
    
    # Find all stitched folders
    stitched_folders = find_stitched_folders(parent_dir)
    
    if not stitched_folders:
        print("No folders with 'stitched' in path containing images were found.")
        return
    
    print(f"Found {len(stitched_folders)} stitched folder(s)")
    print()
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    # Build list of all images to process
    total_processed = 0
    total_failed = 0
    all_tasks = []
    
    print("Preparing image list...")
    for stitched_folder in stitched_folders:
        # Get relative path from parent directory
        rel_path = os.path.relpath(stitched_folder, parent_dir)
        
        # Create corresponding output folder
        output_folder = os.path.join(output_dir, rel_path)
        
        # Collect all images in this folder
        try:
            files = os.listdir(stitched_folder)
            image_files = [f for f in files if Path(f).suffix.lower() in image_extensions]
            
            for image_file in image_files:
                input_path = os.path.join(stitched_folder, image_file)
                
                # Change extension to .jpg for output
                output_filename = Path(image_file).stem + '.jpg'
                output_path = os.path.join(output_folder, output_filename)
                
                all_tasks.append((input_path, output_path, 50))
                
        except Exception as e:
            print(f"Error preparing folder {stitched_folder}: {e}")
    
    total_images = len(all_tasks)
    print(f"Total images to process: {total_images}\n")
    
    if total_images == 0:
        print("No images found to process.")
        return
    
    # Process images in parallel
    start_time = time.time()
    
    try:
        with Pool(num_workers) as pool:
            current = 0
            for result in pool.imap_unordered(compress_image_optimized, all_tasks):
                current += 1
                elapsed_time = time.time() - start_time
                
                success, error_msg = result
                
                if success:
                    total_processed += 1
                else:
                    total_failed += 1
                
                # Print progress bar
                print_progress_bar(current, total_images, elapsed_time, 'Overall Progress')
    
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during processing: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Print newline to move past progress bar
    print()
    
    # Final summary
    total_time = time.time() - start_time
    print("=" * 80)
    print(f"Processing complete!")
    print(f"Total images processed: {total_processed}/{total_images}")
    if total_failed > 0:
        print(f"Failed to process: {total_failed}")
    print(f"Total time: {format_time(total_time)}")
    print(f"Average time per image: {format_time(total_time / total_processed if total_processed > 0 else 0)}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


def main():
    """Main entry point"""
    if len(sys.argv) < 3:
        print("Usage: python image_stitched_processor.py <parent_directory> <output_directory> [num_workers]")
        print()
        print("Arguments:")
        print("  parent_directory   - Directory to search for 'stitched' folders")
        print("  output_directory   - Directory where processed images will be saved")
        print("  num_workers        - (Optional) Number of parallel workers (default: CPU count - 1)")
        print()
        print("Features:")
        print("  - Parallel processing for fast compression")
        print("  - Finds all folders with 'stitched' in the path")
        print("  - Preserves directory structure in output")
        print("  - Reduces image resolution to 50%")
        print("  - Compresses to 50% JPG quality")
        print("  - Strips EXIF metadata")
        sys.exit(1)
    
    parent_dir = sys.argv[1]
    output_dir = sys.argv[2]
    num_workers = None
    
    if len(sys.argv) > 3:
        try:
            num_workers = int(sys.argv[3])
            if num_workers < 1:
                print("Error: num_workers must be at least 1")
                sys.exit(1)
        except ValueError:
            print("Error: num_workers must be an integer")
            sys.exit(1)
    
    process_stitched_images(parent_dir, output_dir, num_workers)


if __name__ == '__main__':
    main()