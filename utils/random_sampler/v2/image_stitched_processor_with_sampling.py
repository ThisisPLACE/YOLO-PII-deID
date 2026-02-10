#!/usr/bin/env python3
"""
Image Stitched Folder Processor - Enhanced Edition with Random Sampling

Advanced image processing with comprehensive options:
- Configurable quality, resolution, and output formats
- Random sampling (sample N images or X% of images)
- Resume capability (skip existing files)
- Dry-run mode for preview
- Advanced filtering (size, dimensions, date, aspect ratio)
- EXIF metadata control (preserve, extract, or strip)
- Memory-aware processing
- Progress logging and manifests
- Single-threaded debugging mode
- Batch processing with presets
"""

import os
import sys
import json
import time
import logging
import random
from pathlib import Path
from datetime import timedelta, datetime
from PIL import Image
from PIL.Image import Resampling
from multiprocessing import Pool, cpu_count
from functools import partial
import traceback
import argparse
from typing import Tuple, List, Dict, Optional
import shutil

# Try to import EXIF handling
try:
    from PIL.Image import Exif
    EXIF_SUPPORT = True
except ImportError:
    EXIF_SUPPORT = False


# ==================== UTILITY FUNCTIONS ====================

def format_time(seconds):
    """Format seconds to human-readable time"""
    return str(timedelta(seconds=int(seconds)))


def format_size(bytes_size):
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}TB"


def setup_logging(log_file=None):
    """Setup logging to console and optionally to file"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[logging.StreamHandler()]
        )


def print_progress_bar(current, total, elapsed_time, prefix='Progress'):
    """
    Print a progress bar with percentage and ETA
    
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
    
    if current > 0 and percent < 1.0:
        processing_rate = current / elapsed_time
        remaining_items = total - current
        eta_seconds = remaining_items / processing_rate
        eta_str = format_time(eta_seconds)
    else:
        eta_str = "calculating..."
    
    elapsed_str = format_time(elapsed_time)
    percentage = percent * 100
    
    print(f'\r{prefix}: |{bar}| {current}/{total} ({percentage:5.1f}%) | Elapsed: {elapsed_str} | ETA: {eta_str}', end='', flush=True)


# ==================== QUALITY PRESETS ====================

QUALITY_PRESETS = {
    'fast': {
        'quality': 40,
        'resize_percent': 50,
        'optimize': False,
        'format': 'JPEG',
        'description': 'Fast compression (40% quality, 50% resolution)'
    },
    'balanced': {
        'quality': 60,
        'resize_percent': 75,
        'optimize': False,
        'format': 'JPEG',
        'description': 'Balanced quality and size (60% quality, 75% resolution)'
    },
    'high': {
        'quality': 80,
        'resize_percent': 100,
        'optimize': False,
        'format': 'JPEG',
        'description': 'High quality (80% quality, original resolution)'
    },
    'lossless': {
        'quality': 95,
        'resize_percent': 100,
        'optimize': False,
        'format': 'PNG',
        'description': 'Lossless compression (95% quality, original resolution, PNG)'
    },
    'webp-fast': {
        'quality': 50,
        'resize_percent': 50,
        'optimize': False,
        'format': 'WEBP',
        'description': 'Fast WebP compression (50% quality, 50% resolution)'
    },
    'webp-balanced': {
        'quality': 70,
        'resize_percent': 75,
        'optimize': False,
        'format': 'WEBP',
        'description': 'Balanced WebP (70% quality, 75% resolution)'
    }
}


# ==================== RANDOM SAMPLING ====================

def apply_random_sampling(image_files, sample_size=None, sample_percent=None, seed=None):
    """
    Apply random sampling to image list
    
    Args:
        image_files (list): List of (input_path, rel_path, filename) tuples
        sample_size (int): Exact number of images to sample
        sample_percent (float): Percentage of images to sample (0-100)
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (sampled_files, sampling_info_dict)
    """
    if not image_files:
        return image_files, {'total': 0, 'sampled': 0, 'percent': 0}
    
    total = len(image_files)
    
    # Set seed for reproducibility
    if seed is not None:
        random.seed(seed)
        logging.info(f"Random seed: {seed}")
    
    # Determine actual sample size
    if sample_size is not None:
        actual_size = min(sample_size, total)
        percent = (actual_size / total * 100) if total > 0 else 0
    elif sample_percent is not None:
        actual_size = max(1, int(total * sample_percent / 100))
        percent = sample_percent
    else:
        return image_files, {'total': total, 'sampled': total, 'percent': 100}
    
    # Perform sampling
    sampled = random.sample(image_files, actual_size)
    
    sampling_info = {
        'total': total,
        'sampled': actual_size,
        'percent': percent
    }
    
    return sampled, sampling_info


# ==================== IMAGE PROCESSING ====================

def extract_exif_data(image_path):
    """
    Extract EXIF data from image
    
    Returns:
        dict: EXIF data or empty dict if none
    """
    try:
        if not EXIF_SUPPORT:
            return {}
        
        img = Image.open(image_path)
        exif_data = img.getexif()
        
        if not exif_data:
            return {}
        
        # Convert to serializable format
        exif_dict = {}
        for tag, value in exif_data.items():
            try:
                exif_dict[str(tag)] = str(value)
            except:
                pass
        
        return exif_dict
    except Exception as e:
        logging.debug(f"Could not extract EXIF from {image_path}: {e}")
        return {}


def process_image(input_path, output_path, config, preserve_exif=False, 
                  extract_exif_path=None, dry_run=False, single_threaded=False):
    """
    Process a single image with comprehensive options
    
    Args:
        input_path (str): Input image path
        output_path (str): Output image path
        config (dict): Processing configuration
        preserve_exif (bool): Preserve EXIF data in output
        extract_exif_path (str): Path to save extracted EXIF as JSON
        dry_run (bool): Don't actually save, just validate
        single_threaded (bool): Enable debug logging
        
    Returns:
        tuple: (success, error_msg, stats_dict)
    """
    stats = {
        'input_size': 0,
        'output_size': 0,
        'compression_ratio': 0,
        'original_dims': None,
        'output_dims': None,
        'processing_time': 0
    }
    
    try:
        start = time.time()
        
        # Get input file size
        input_size = os.path.getsize(input_path)
        stats['input_size'] = input_size
        
        # Open image
        img = Image.open(input_path)
        original_dims = (img.width, img.height)
        stats['original_dims'] = original_dims
        
        if single_threaded:
            logging.debug(f"Processing: {input_path}")
            logging.debug(f"  Original dimensions: {original_dims}")
        
        # Extract EXIF if requested
        exif_data = {}
        if EXIF_SUPPORT and extract_exif_path:
            exif_data = extract_exif_data(input_path)
        
        # Convert color space
        if img.mode != 'RGB':
            if img.mode == 'RGBA':
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
        
        # Resize if specified
        resize_percent = config.get('resize_percent', 100)
        if resize_percent < 100:
            new_width = int(img.width * resize_percent / 100)
            new_height = int(img.height * resize_percent / 100)
            img = img.resize((new_width, new_height), Resampling.LANCZOS)
        
        stats['output_dims'] = (img.width, img.height)
        
        # Don't actually save in dry-run mode
        if dry_run:
            stats['processing_time'] = time.time() - start
            return (True, None, stats)
        
        # Create output directory
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save image
        save_kwargs = {
            'quality': config.get('quality', 85),
            'optimize': config.get('optimize', False)
        }
        
        # Handle EXIF preservation
        if preserve_exif and EXIF_SUPPORT:
            try:
                original_img = Image.open(input_path)
                exif = original_img.getexif()
                if exif:
                    save_kwargs['exif'] = exif
            except:
                pass
        
        img.save(output_path, config.get('format', 'JPEG'), **save_kwargs)
        
        # Get output file size
        output_size = os.path.getsize(output_path)
        stats['output_size'] = output_size
        stats['compression_ratio'] = (1 - output_size / input_size) * 100 if input_size > 0 else 0
        
        # Save extracted EXIF if requested
        if extract_exif_path and exif_data:
            os.makedirs(os.path.dirname(extract_exif_path), exist_ok=True)
            with open(extract_exif_path, 'w') as f:
                json.dump({
                    'filename': os.path.basename(input_path),
                    'timestamp': datetime.now().isoformat(),
                    'exif_data': exif_data
                }, f, indent=2)
        
        stats['processing_time'] = time.time() - start
        return (True, None, stats)
        
    except Exception as e:
        stats['processing_time'] = time.time() - start
        return (False, str(e), stats)


def compress_image_worker(args):
    """Worker function for multiprocessing"""
    (input_path, output_path, config, preserve_exif, extract_exif_path, dry_run) = args
    return process_image(input_path, output_path, config, preserve_exif, extract_exif_path, dry_run)


# ==================== FILE FILTERING ====================

def passes_filters(file_path, filters):
    """
    Check if file passes all filters
    
    Args:
        file_path (str): Path to file
        filters (dict): Filter configuration
        
    Returns:
        bool: True if file passes all filters
    """
    try:
        # File type filter
        if filters.get('file_types'):
            ext = Path(file_path).suffix.lower()
            if ext not in filters['file_types']:
                return False
        
        # File size filter
        file_size = os.path.getsize(file_path)
        if filters.get('min_size') and file_size < filters['min_size']:
            return False
        if filters.get('max_size') and file_size > filters['max_size']:
            return False
        
        # Date filter
        if filters.get('modified_after'):
            mod_time = os.path.getmtime(file_path)
            if mod_time < filters['modified_after']:
                return False
        
        if filters.get('modified_before'):
            mod_time = os.path.getmtime(file_path)
            if mod_time > filters['modified_before']:
                return False
        
        # Dimension filters
        if filters.get('min_width') or filters.get('min_height') or \
           filters.get('max_width') or filters.get('max_height') or \
           filters.get('aspect_ratio'):
            try:
                img = Image.open(file_path)
                width, height = img.size
                
                if filters.get('min_width') and width < filters['min_width']:
                    return False
                if filters.get('max_width') and width > filters['max_width']:
                    return False
                if filters.get('min_height') and height < filters['min_height']:
                    return False
                if filters.get('max_height') and height > filters['max_height']:
                    return False
                
                # Aspect ratio check
                if filters.get('aspect_ratio'):
                    min_ar, max_ar = filters['aspect_ratio']
                    current_ar = width / height if height > 0 else 0
                    if not (min_ar <= current_ar <= max_ar):
                        return False
            except:
                return False
        
        return True
        
    except Exception as e:
        logging.debug(f"Filter check failed for {file_path}: {e}")
        return False


# ==================== DIRECTORY SCANNING ====================

def find_image_files(parent_dir, stitched_only=True, recursive=True, exclude_dirs=None):
    """
    Find all image files in directory
    
    Args:
        parent_dir (str): Parent directory
        stitched_only (bool): Only find folders with "stitched" in path
        recursive (bool): Search recursively
        exclude_dirs (set): Directories to exclude
        
    Returns:
        list: List of image file paths with relative paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_files = []
    exclude_dirs = exclude_dirs or set()
    
    try:
        for root, dirs, files in os.walk(parent_dir):
            # Remove excluded directories from dirs to prevent os.walk from traversing them
            dirs[:] = [d for d in dirs if d not in exclude_dirs and os.path.join(root, d) not in exclude_dirs]
            
            # Check depth if not recursive
            if not recursive:
                depth = root[len(parent_dir):].count(os.sep)
                if depth > 0:
                    dirs.clear()
                    continue
            
            # Check if "stitched" filter applies
            if stitched_only and 'stitched' not in root.lower():
                continue
            
            # Collect image files
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(root, parent_dir)
                    image_files.append((full_path, rel_path, file))
    
    except Exception as e:
        logging.error(f"Error scanning directory: {e}")
        return []
    
    return image_files


# ==================== MANIFEST GENERATION ====================

def create_manifest(output_file, tasks_completed, config, start_time, end_time, filters, sampling_info=None):
    """
    Create a processing manifest/report
    
    Args:
        output_file (str): Path to manifest file
        tasks_completed (list): List of completed task results
        config (dict): Processing configuration
        start_time (float): Start timestamp
        end_time (float): End timestamp
        filters (dict): Applied filters
        sampling_info (dict): Random sampling information
    """
    total_time = end_time - start_time
    total_input = sum(t.get('input_size', 0) for t in tasks_completed)
    total_output = sum(t.get('output_size', 0) for t in tasks_completed)
    
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'processing_duration': format_time(total_time),
        'total_duration_seconds': total_time,
        'config': config,
        'filters': filters,
        'summary': {
            'total_files': len(tasks_completed),
            'total_input_size': format_size(total_input),
            'total_output_size': format_size(total_output),
            'total_compression': (1 - total_output / total_input) * 100 if total_input > 0 else 0,
            'average_processing_time_per_file': format_time(total_time / len(tasks_completed) if tasks_completed else 0)
        },
        'files': []
    }
    
    # Add sampling info if present
    if sampling_info:
        manifest['sampling'] = {
            'total_available': sampling_info['total'],
            'sampled': sampling_info['sampled'],
            'sample_percent': sampling_info['percent']
        }
    
    for task in tasks_completed:
        manifest['files'].append({
            'compression_ratio': f"{task.get('compression_ratio', 0):.1f}%",
            'input_size': format_size(task.get('input_size', 0)),
            'output_size': format_size(task.get('output_size', 0)),
            'original_dimensions': task.get('original_dims'),
            'output_dimensions': task.get('output_dims'),
            'processing_time': format_time(task.get('processing_time', 0))
        })
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(manifest, f, indent=2)


# ==================== MAIN PROCESSING ====================

def process_stitched_images(args):
    """
    Main processing function with all options
    
    Args:
        args: Parsed command line arguments
    """
    # Validate inputs
    if not os.path.isdir(args.parent_dir):
        logging.error(f"Parent directory '{args.parent_dir}' does not exist")
        sys.exit(1)
    
    # Setup logging
    log_file = None
    if args.log_file:
        log_file = os.path.join(args.output_dir, args.log_file) if not os.path.isabs(args.log_file) else args.log_file
    setup_logging(log_file)
    
    # Create output directory
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except Exception as e:
        logging.error(f"Error creating output directory: {e}")
        sys.exit(1)
    
    # Get processing configuration
    if args.preset:
        if args.preset not in QUALITY_PRESETS:
            logging.error(f"Unknown preset: {args.preset}")
            logging.info(f"Available presets: {', '.join(QUALITY_PRESETS.keys())}")
            sys.exit(1)
        config = QUALITY_PRESETS[args.preset].copy()
    else:
        config = {
            'quality': args.quality,
            'resize_percent': args.resize_percent,
            'optimize': args.optimize,
            'format': args.output_format.upper()
        }
    
    # Build filters
    filters = {
        'file_types': set(args.file_types) if args.file_types else {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'},
        'min_size': args.min_size,
        'max_size': args.max_size,
        'min_width': args.min_width,
        'max_width': args.max_width,
        'min_height': args.min_height,
        'max_height': args.max_height,
    }
    
    if args.aspect_ratio:
        parts = args.aspect_ratio.split(',')
        filters['aspect_ratio'] = (float(parts[0]), float(parts[1]))
    
    if args.modified_after:
        filters['modified_after'] = datetime.fromisoformat(args.modified_after).timestamp()
    
    if args.modified_before:
        filters['modified_before'] = datetime.fromisoformat(args.modified_before).timestamp()
    
    # Build exclusion set
    exclude_dirs = set(args.exclude_dirs) if args.exclude_dirs else set()
    
    # Print configuration
    logging.info("=" * 80)
    logging.info("Image Processing Configuration")
    logging.info("=" * 80)
    logging.info(f"Parent directory: {args.parent_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Processing mode: {'DRY RUN (preview only)' if args.dry_run else 'NORMAL'}")
    logging.info(f"Stitched folders only: {not args.all_images}")
    logging.info(f"Recursive search: {args.recursive}")
    logging.info(f"Quality preset: {args.preset or 'Custom'}")
    logging.info(f"Output format: {config['format']}")
    logging.info(f"Quality: {config['quality']}")
    logging.info(f"Resize: {config['resize_percent']}%")
    logging.info(f"Workers: {args.num_workers if not args.single_thread else 'Single-threaded (debug mode)'}")
    logging.info(f"Preserve EXIF: {args.preserve_exif}")
    logging.info(f"Skip existing files: {args.resume}")
    
    # Log sampling info
    if args.sample_size:
        logging.info(f"Random sampling: {args.sample_size} images")
    elif args.sample_percent:
        logging.info(f"Random sampling: {args.sample_percent}% of images")
    
    logging.info("")
    
    # Find images
    logging.info("Scanning for images...")
    image_files = find_image_files(
        args.parent_dir,
        stitched_only=not args.all_images,
        recursive=args.recursive,
        exclude_dirs=exclude_dirs
    )
    
    if not image_files:
        logging.warning("No images found to process.")
        return
    
    logging.info(f"Found {len(image_files)} image files")
    
    # Apply random sampling
    sampling_info = None
    if args.sample_size or args.sample_percent:
        image_files, sampling_info = apply_random_sampling(
            image_files,
            sample_size=args.sample_size,
            sample_percent=args.sample_percent,
            seed=args.random_seed
        )
        logging.info(f"Random sampling: {sampling_info['sampled']} images selected ({sampling_info['percent']:.1f}%)")
    
    # Build processing tasks
    all_tasks = []
    skipped_resume = 0
    skipped_filters = 0
    
    logging.info("Preparing task list...")
    
    for input_path, rel_path, filename in image_files:
        # Apply filters
        if not passes_filters(input_path, filters):
            skipped_filters += 1
            continue
        
        # Preserve original filename by default unless converting format
        if args.preserve_filename:
            output_filename = filename
        else:
            output_filename = Path(filename).stem + '.' + config['format'].lower()
        
        # Apply prefix/suffix
        if args.filename_prefix or args.filename_suffix:
            stem = Path(output_filename).stem
            ext = Path(output_filename).suffix
            output_filename = f"{args.filename_prefix}{stem}{args.filename_suffix}{ext}"
        
        # Create output path
        output_folder = os.path.join(args.output_dir, rel_path)
        output_path = os.path.join(output_folder, output_filename)
        
        # Check if should skip (resume mode)
        if args.resume and os.path.exists(output_path):
            skipped_resume += 1
            continue
        
        # Prepare EXIF paths if needed
        extract_exif_path = None
        if args.extract_exif:
            exif_dir = os.path.join(args.output_dir, 'exif_data', rel_path)
            extract_exif_path = os.path.join(exif_dir, Path(filename).stem + '.json')
        
        all_tasks.append((input_path, output_path, config, args.preserve_exif, extract_exif_path, args.dry_run))
    
    total_images = len(all_tasks)
    
    logging.info(f"Tasks prepared: {total_images}")
    if skipped_resume > 0:
        logging.info(f"Skipped (already exist): {skipped_resume}")
    if skipped_filters > 0:
        logging.info(f"Skipped (filters): {skipped_filters}")
    logging.info("")
    
    if total_images == 0:
        logging.warning("No images to process after filtering.")
        return
    
    if args.dry_run:
        logging.warning("DRY RUN MODE: No files will actually be modified")
        logging.info("")
    
    # Process images
    start_time = time.time()
    total_processed = 0
    total_failed = 0
    completed_tasks = []
    
    try:
        if args.single_thread:
            # Single-threaded mode for debugging
            logging.info("Processing in single-threaded mode...")
            for idx, task in enumerate(all_tasks, 1):
                input_path, output_path, config, preserve_exif, extract_exif_path, dry_run = task
                success, error_msg, stats = process_image(
                    input_path, output_path, config, preserve_exif, extract_exif_path, dry_run, single_threaded=True
                )
                
                if success:
                    total_processed += 1
                    completed_tasks.append(stats)
                else:
                    total_failed += 1
                    logging.error(f"Failed to process {input_path}: {error_msg}")
                
                elapsed_time = time.time() - start_time
                print_progress_bar(idx, total_images, elapsed_time, 'Progress')
        else:
            # Multi-threaded mode
            num_workers = args.num_workers or max(1, cpu_count() - 1)
            logging.info(f"Processing with {num_workers} workers...")
            
            with Pool(num_workers) as pool:
                current = 0
                for result in pool.imap_unordered(compress_image_worker, all_tasks):
                    current += 1
                    elapsed_time = time.time() - start_time
                    
                    success, error_msg, stats = result
                    
                    if success:
                        total_processed += 1
                        completed_tasks.append(stats)
                    else:
                        total_failed += 1
                    
                    print_progress_bar(current, total_images, elapsed_time, 'Progress')
    
    except KeyboardInterrupt:
        logging.warning("\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print()  # Newline after progress bar
    
    # Generate manifest if requested
    end_time = time.time()
    if args.manifest:
        manifest_path = os.path.join(args.output_dir, args.manifest)
        logging.info(f"Generating manifest: {manifest_path}")
        create_manifest(manifest_path, completed_tasks, config, start_time, end_time, filters, sampling_info)
    
    # Print summary
    total_time = end_time - start_time
    logging.info("=" * 80)
    logging.info("Processing Complete!")
    logging.info("=" * 80)
    logging.info(f"Total processed: {total_processed}/{total_images}")
    if total_failed > 0:
        logging.info(f"Failed: {total_failed}")
    logging.info(f"Total time: {format_time(total_time)}")
    if total_processed > 0:
        logging.info(f"Average time per image: {format_time(total_time / total_processed)}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info("=" * 80)


# ==================== COMMAND LINE INTERFACE ====================

def create_parser():
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description='Advanced image stitching processor with random sampling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Random sample: process 100 random images
  python script.py /input /output --sample-size 100
  
  # Random sample: process 10% of images
  python script.py /input /output --sample-percent 10
  
  # Reproducible sampling: same 100 images every time
  python script.py /input /output --sample-size 100 --random-seed 42
  
  # Quality preset
  python script.py /input /output --preset balanced
  
  # Process all images (not just stitched)
  python script.py /input /output --all-images
  
  # Resume interrupted processing
  python script.py /input /output --resume
  
  # Dry run to preview
  python script.py /input /output --dry-run
  
  # Extract EXIF data and preserve it
  python script.py /input /output --extract-exif --preserve-exif
  
  # Advanced filtering
  python script.py /input /output --min-size 1MB --max-size 50MB --min-width 1920
  
  # Generate processing report
  python script.py /input /output --manifest processing_report.json
        """
    )
    
    # Required arguments
    parser.add_argument('parent_dir', help='Parent directory to search for images')
    parser.add_argument('output_dir', help='Output directory for processed images')
    
    # Random sampling options
    parser.add_argument('--sample-size', type=int,
                        help='Process only N random images (e.g., 100)')
    parser.add_argument('--sample-percent', type=float,
                        help='Process only X%% of images (e.g., 10 for 10%%)')
    parser.add_argument('--random-seed', type=int,
                        help='Seed for random sampling (for reproducibility)')
    
    # Quality and format options
    parser.add_argument('--preset', choices=list(QUALITY_PRESETS.keys()),
                        help='Use a quality preset')
    parser.add_argument('--quality', type=int, default=60,
                        help='Output quality (1-100, default: 60)')
    parser.add_argument('--resize', dest='resize_percent', type=int, default=75,
                        help='Resize to percentage of original (default: 75)')
    parser.add_argument('--format', dest='output_format', default='JPEG',
                        choices=['JPEG', 'PNG', 'WEBP', 'BMP'],
                        help='Output format (default: JPEG)')
    parser.add_argument('--optimize', action='store_true',
                        help='Enable PIL image optimization (slower but smaller)')
    parser.add_argument('--preserve-filename', action='store_true',
                        help='Keep original filename (ignore format changes)')
    parser.add_argument('--filename-prefix', default='',
                        help='Add prefix to output filenames')
    parser.add_argument('--filename-suffix', default='',
                        help='Add suffix to output filenames')
    
    # Processing options
    parser.add_argument('--all-images', action='store_true',
                        help='Process all images (not just stitched folders)')
    parser.add_argument('--recursive', action='store_true', default=True,
                        help='Search recursively (default: True)')
    parser.add_argument('--no-recursive', dest='recursive', action='store_false',
                        help='Do not search recursively')
    parser.add_argument('--exclude-dirs', nargs='*', default=[],
                        help='Directories to exclude from processing')
    parser.add_argument('--resume', action='store_true',
                        help='Skip files that already exist in output')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview what would be processed without saving')
    
    # Filtering options
    parser.add_argument('--file-types', nargs='*',
                        help='File types to process (e.g., .jpg .png)')
    parser.add_argument('--min-size', type=str,
                        help='Minimum file size (e.g., 1MB, 500KB)')
    parser.add_argument('--max-size', type=str,
                        help='Maximum file size (e.g., 50MB, 100MB)')
    parser.add_argument('--min-width', type=int,
                        help='Minimum image width in pixels')
    parser.add_argument('--max-width', type=int,
                        help='Maximum image width in pixels')
    parser.add_argument('--min-height', type=int,
                        help='Minimum image height in pixels')
    parser.add_argument('--max-height', type=int,
                        help='Maximum image height in pixels')
    parser.add_argument('--aspect-ratio', type=str,
                        help='Aspect ratio range (e.g., "0.5,2.0" for 1:2 to 2:1)')
    parser.add_argument('--modified-after', type=str,
                        help='Only process files modified after date (YYYY-MM-DD)')
    parser.add_argument('--modified-before', type=str,
                        help='Only process files modified before date (YYYY-MM-DD)')
    
    # EXIF options
    parser.add_argument('--preserve-exif', action='store_true',
                        help='Preserve EXIF data in output images')
    parser.add_argument('--extract-exif', action='store_true',
                        help='Extract EXIF data to JSON files')
    
    # Performance options
    parser.add_argument('--num-workers', type=int,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--single-thread', action='store_true',
                        help='Process single-threaded (debug mode)')
    
    # Logging and reporting options
    parser.add_argument('--log-file', type=str,
                        help='Log file path (default: stdout only)')
    parser.add_argument('--manifest', type=str,
                        help='Generate processing manifest JSON file')
    
    return parser


def parse_size(size_str):
    """Parse size string like '1MB' or '500KB' to bytes"""
    if not size_str:
        return None
    
    units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    size_str = size_str.strip().upper()
    
    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            try:
                value = float(size_str[:-len(unit)])
                return int(value * multiplier)
            except ValueError:
                return None
    
    try:
        return int(size_str)
    except ValueError:
        return None


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Parse size arguments
    if args.min_size:
        args.min_size = parse_size(args.min_size)
    if args.max_size:
        args.max_size = parse_size(args.max_size)
    
    # Convert file types to lowercase with dots
    if args.file_types:
        args.file_types = [ft if ft.startswith('.') else f'.{ft}' for ft in args.file_types]
        args.file_types = [ft.lower() for ft in args.file_types]
    
    # Validate sampling options
    if args.sample_size and args.sample_percent:
        print("Error: Cannot use both --sample-size and --sample-percent together")
        sys.exit(1)
    
    if args.sample_size and args.sample_size < 1:
        print("Error: --sample-size must be at least 1")
        sys.exit(1)
    
    if args.sample_percent:
        if args.sample_percent < 0.1 or args.sample_percent > 100:
            print("Error: --sample-percent must be between 0.1 and 100")
            sys.exit(1)
    
    process_stitched_images(args)


if __name__ == '__main__':
    main()
