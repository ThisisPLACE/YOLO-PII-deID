#!/usr/bin/env python3
"""
Select a random sample of images from a directory and its subdirectories.

This script recursively searches through a directory tree, finds all image files,
and copies a random sample to an output directory while preserving subdirectory structure
or flattening as desired.

Supported image formats: jpg, jpeg, png, bmp, gif, tiff, webp

Usage:
    python select_random_images.py -i <input_dir> -o <output_dir> -n <num_images> \
        [--preserve-structure] [--seed 42]
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from typing import List, Tuple

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


def find_all_images(root_dir: str) -> List[Tuple[str, str]]:
    """
    Recursively find all image files in directory tree.
    
    Returns a list of (full_path, relative_path) tuples for each image found.
    """
    images = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_ext = Path(filename).suffix.lower()
            
            if file_ext in IMAGE_EXTENSIONS:
                full_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(full_path, root_dir)
                images.append((full_path, relative_path))
    
    return images


def select_random_sample(images: List[Tuple[str, str]], n: int, seed: int = None) -> List[Tuple[str, str]]:
    """
    Select n random images from the list.
    """
    if seed is not None:
        random.seed(seed)
    
    if n > len(images):
        print(f"Warning: Requested {n} images but only {len(images)} available.")
        print(f"Selecting all {len(images)} images instead.\n")
        return images
    
    return random.sample(images, n)


def copy_images(selected_images: List[Tuple[str, str]], output_dir: str, 
                preserve_structure: bool = False) -> None:
    """
    Copy selected images to output directory.
    
    If preserve_structure is True, maintains the original subdirectory structure.
    Otherwise, flattens all images into the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    copied = 0
    
    for full_path, relative_path in selected_images:
        if preserve_structure:
            # Maintain subdirectory structure
            output_path = os.path.join(output_dir, relative_path)
            output_subdir = os.path.dirname(output_path)
            os.makedirs(output_subdir, exist_ok=True)
        else:
            # Flatten into single directory
            filename = os.path.basename(full_path)
            output_path = os.path.join(output_dir, filename)
            
            # Handle name conflicts by adding counter
            counter = 1
            base_name = Path(filename).stem
            ext = Path(filename).suffix
            
            while os.path.exists(output_path):
                filename = f"{base_name}_{counter}{ext}"
                output_path = os.path.join(output_dir, filename)
                counter += 1
        
        try:
            shutil.copy2(full_path, output_path)
            copied += 1
        except Exception as e:
            print(f"Error copying {full_path}: {e}")
    
    return copied


def print_summary(input_dir: str, output_dir: str, total_found: int, 
                 selected: int, copied: int, preserve_structure: bool) -> None:
    """
    Print summary of the operation.
    """
    print("\n" + "="*60)
    print("RANDOM IMAGE SELECTION COMPLETE")
    print("="*60)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"\nImages found:   {total_found}")
    print(f"Images selected: {selected}")
    print(f"Images copied:   {copied}")
    print(f"\nStructure: {'Preserved' if preserve_structure else 'Flattened'}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Select random images from a directory and its subdirectories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Select 50 random images, flatten structure
  python select_random_images.py -i /path/to/images -o /path/to/output -n 50
  
  # Select 200 images, preserve subdirectory structure
  python select_random_images.py -i /path/to/images -o /path/to/output -n 200 \\
      --preserve-structure
  
  # Use fixed seed for reproducibility
  python select_random_images.py -i /path/to/images -o /path/to/output -n 100 \\
      --seed 42
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Input directory containing images in subdirectories')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory for selected images')
    parser.add_argument('-n', '--number', type=int, required=True,
                       help='Number of random images to select')
    parser.add_argument('--preserve-structure', action='store_true',
                       help='Preserve original subdirectory structure (default: flatten)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        return 1
    
    # Validate number
    if args.number <= 0:
        print(f"Error: Number of images must be positive, got {args.number}")
        return 1
    
    print(f"Searching for images in: {args.input}")
    
    # Find all images
    all_images = find_all_images(args.input)
    
    if not all_images:
        print("Error: No images found in the directory tree")
        return 1
    
    print(f"Found {len(all_images)} total images\n")
    
    # Select random sample
    print(f"Selecting {args.number} random images...")
    selected_images = select_random_sample(all_images, args.number, args.seed)
    
    # Copy images to output
    print(f"Copying images to: {args.output}")
    copied = copy_images(selected_images, args.output, args.preserve_structure)
    
    # Print summary
    print_summary(args.input, args.output, len(all_images), 
                 len(selected_images), copied, args.preserve_structure)
    
    return 0


if __name__ == '__main__':
    exit(main())
