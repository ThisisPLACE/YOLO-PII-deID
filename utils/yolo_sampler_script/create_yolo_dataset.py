#!/usr/bin/env python3
"""
Create a YOLO training dataset from a representative sample of images.

This script recursively searches through subdirectories, identifies images and their
corresponding annotation files (sidecar .txt files with _detect suffix), and creates a 
properly structured YOLO dataset with train/val/test splits and a ready-to-use data.yaml 
configuration file.

The output follows YOLO's expected directory structure:
    dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml

Usage:
    python create_yolo_dataset.py -i <input_dir> -o <output_dir> [-n <num_images>] \
        [--train 0.7] [--val 0.2] [--test 0.1]
"""

import os
import shutil
import argparse
import random
import yaml
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}


def find_images_and_annotations(root_dir: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Recursively find all images and their corresponding annotation files.
    
    Returns a dictionary mapping subdirectory paths to lists of (image_path, annotation_path) tuples.
    """
    images_by_dir = defaultdict(list)
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Find all image files in this directory
        for filename in filenames:
            file_ext = Path(filename).suffix.lower()
            
            if file_ext in IMAGE_EXTENSIONS:
                image_path = os.path.join(dirpath, filename)
                
                # Look for corresponding annotation file (same name with _detect suffix, .txt extension)
                annotation_name = Path(filename).stem + '.txt'
                annotation_path = os.path.join(dirpath, annotation_name)
                
                # Only include if annotation file exists
                if os.path.exists(annotation_path):
                    images_by_dir[dirpath].append((image_path, annotation_path))
    
    return images_by_dir


def get_stratified_sample(images_by_dir: Dict[str, List[Tuple[str, str]]], 
                         total_samples: int) -> List[Tuple[str, str]]:
    """
    Get a stratified random sample of (image, annotation) pairs.
    
    The sample is stratified by source directory to ensure representation
    proportional to the number of images in each directory.
    """
    # Calculate total images and proportions
    total_images = sum(len(pairs) for pairs in images_by_dir.values())
    
    if total_images == 0:
        raise ValueError("No images with annotations found in the directory tree")
    
    if total_samples > total_images:
        print(f"Warning: Requested {total_samples} images but only {total_images} available.")
        print(f"Using all {total_images} images instead.")
        total_samples = total_images
    
    # Calculate how many samples to take from each directory
    samples_per_dir = {}
    remaining_samples = total_samples
    
    for dirpath in sorted(images_by_dir.keys()):
        dir_count = len(images_by_dir[dirpath])
        proportion = dir_count / total_images
        samples_needed = round(proportion * total_samples)
        samples_needed = min(samples_needed, dir_count)  # Can't take more than available
        samples_per_dir[dirpath] = samples_needed
        remaining_samples -= samples_needed
    
    # Distribute any remaining samples due to rounding
    if remaining_samples > 0:
        dirs_with_capacity = [
            dirpath for dirpath, count in samples_per_dir.items()
            if count < len(images_by_dir[dirpath])
        ]
        for dirpath in dirs_with_capacity[:remaining_samples]:
            samples_per_dir[dirpath] += 1
    
    # Select random samples from each directory
    selected_pairs = []
    
    for dirpath, num_samples in samples_per_dir.items():
        if num_samples > 0:
            pairs = images_by_dir[dirpath]
            sampled = random.sample(pairs, num_samples)
            selected_pairs.extend(sampled)
            print(f"Selected {num_samples}/{len(pairs)} images from: {dirpath}")
    
    return selected_pairs


def extract_classes_from_annotations(selected_pairs: List[Tuple[str, str]]) -> Dict[int, str]:
    """
    Extract unique class IDs from annotations and return a mapping.
    Attempts to preserve original class names if they exist in a classes.txt file.
    """
    class_ids = set()
    
    # Find all unique class IDs
    for _, annotation_path in selected_pairs:
        try:
            with open(annotation_path, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_ids.add(class_id)
        except (ValueError, IndexError):
            continue
    
    # Try to find class names from a classes.txt or similar file
    class_names = {}
    for class_id in sorted(class_ids):
        class_names[class_id] = f"class_{class_id}"
    
    return class_names


def split_into_sets(selected_pairs: List[Tuple[str, str]], 
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.2,
                   test_ratio: float = 0.1) -> Tuple[List, List, List]:
    """
    Split samples into train, validation, and test sets.
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 < total_ratio < 1.01):  # Allow small floating point error
        raise ValueError(f"Train/val/test ratios must sum to 1.0, got {total_ratio}")
    
    # Shuffle the pairs
    shuffled = selected_pairs.copy()
    random.shuffle(shuffled)
    
    # Calculate split indices
    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    
    train_set = shuffled[:train_count]
    val_set = shuffled[train_count:train_count + val_count]
    test_set = shuffled[train_count + val_count:]
    
    return train_set, val_set, test_set


def create_dataset_structure(output_dir: str, class_names: Dict[int, str],
                            train_set: List, val_set: List, test_set: List) -> None:
    """
    Create the YOLO dataset directory structure and copy files.
    """
    # Create directory structure
    splits = {
        'train': train_set,
        'val': val_set,
        'test': test_set
    }
    
    for split_name, pairs in splits.items():
        images_dir = os.path.join(output_dir, 'images', split_name)
        labels_dir = os.path.join(output_dir, 'labels', split_name)
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Copy files for this split
        for image_path, annotation_path in pairs:
            image_filename = os.path.basename(image_path)
            annotation_filename = os.path.basename(annotation_path)
            
            # Handle naming conflicts
            output_image = os.path.join(images_dir, image_filename)
            output_annotation = os.path.join(labels_dir, annotation_filename)
            
            counter = 1
            base_name = Path(image_filename).stem
            ext = Path(image_filename).suffix
            
            while os.path.exists(output_image):
                image_filename = f"{base_name}_{counter}{ext}"
                output_image = os.path.join(images_dir, image_filename)
                annotation_filename = f"{base_name}_{counter}.txt"
                output_annotation = os.path.join(labels_dir, annotation_filename)
                counter += 1
            
            # Copy files
            shutil.copy2(image_path, output_image)
            shutil.copy2(annotation_path, output_annotation)
        
        print(f"Copied {len(pairs)} {split_name} images")


def create_yaml_config(output_dir: str, class_names: Dict[int, str],
                      train_images: int, val_images: int, test_images: int) -> None:
    """
    Create the data.yaml configuration file for YOLO training.
    """
    # Create class list in order
    classes = [class_names[i] for i in sorted(class_names.keys())]
    
    # Create YAML content
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(classes),
        'names': classes
    }
    
    # Write YAML file
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nCreated data.yaml configuration file")
    print(f"  Classes: {len(classes)}")
    print(f"  Train images: {train_images}")
    print(f"  Val images: {val_images}")
    print(f"  Test images: {test_images}")


def print_dataset_summary(output_dir: str, train_set: List, val_set: List, test_set: List) -> None:
    """
    Print a summary of the created dataset.
    """
    total = len(train_set) + len(val_set) + len(test_set)
    
    print("\n" + "="*60)
    print("YOLO DATASET CREATED SUCCESSFULLY")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_set)} images ({100*len(train_set)/total:.1f}%)")
    print(f"  Val:   {len(val_set)} images ({100*len(val_set)/total:.1f}%)")
    print(f"  Test:  {len(test_set)} images ({100*len(test_set)/total:.1f}%)")
    print(f"  Total: {total} images")
    print(f"\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"  ├── images/")
    print(f"  │   ├── train/")
    print(f"  │   ├── val/")
    print(f"  │   └── test/")
    print(f"  ├── labels/")
    print(f"  │   ├── train/")
    print(f"  │   ├── val/")
    print(f"  │   └── test/")
    print(f"  └── data.yaml")
    print(f"\nYAML config ready at: {os.path.join(output_dir, 'data.yaml')}")
    print("\nUsage with YOLOv8:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('yolov8n.pt')")
    print(f"  results = model.train(data='{os.path.join(output_dir, 'data.yaml')}', epochs=100)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Create a YOLO training dataset with train/val/test splits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create dataset with 200 images (default 70/20/10 split)
  python create_yolo_dataset.py -i /path/to/images -o /path/to/dataset
  
  # Create dataset with 500 images and custom split
  python create_yolo_dataset.py -i /path/to/images -o /path/to/dataset \\
      -n 500 --train 0.8 --val 0.15 --test 0.05
  
  # Use fixed seed for reproducibility
  python create_yolo_dataset.py -i /path/to/images -o /path/to/dataset \\
      -n 200 --seed 42
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Input directory containing images in subdirectories')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory for the YOLO dataset')
    parser.add_argument('-n', '--number', type=int, default=200,
                       help='Number of images to sample (default: 200)')
    parser.add_argument('--train', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val', type=float, default=0.2,
                       help='Validation set ratio (default: 0.2)')
    parser.add_argument('--test', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    # Validate input directory
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        return 1
    
    # Validate split ratios
    total_ratio = args.train + args.val + args.test
    if not (0.99 < total_ratio < 1.01):
        print(f"Error: Train/val/test ratios must sum to 1.0, got {total_ratio}")
        return 1
    
    print(f"Searching for images in: {args.input}")
    
    # Find all images and annotations
    images_by_dir = find_images_and_annotations(args.input)
    
    if not images_by_dir:
        print("Error: No images with annotation files found")
        return 1
    
    total_found = sum(len(pairs) for pairs in images_by_dir.values())
    print(f"Found {total_found} images with annotations across {len(images_by_dir)} directories\n")
    
    # Get stratified sample
    print(f"Sampling {args.number} representative images...\n")
    selected_pairs = get_stratified_sample(images_by_dir, args.number)
    
    # Extract class information
    class_names = extract_classes_from_annotations(selected_pairs)
    print(f"\nIdentified {len(class_names)} classes")
    
    # Split into train/val/test
    print(f"\nSplitting into train ({args.train*100:.0f}%) / val ({args.val*100:.0f}%) / test ({args.test*100:.0f}%)")
    train_set, val_set, test_set = split_into_sets(
        selected_pairs, 
        args.train, 
        args.val, 
        args.test
    )
    
    # Create dataset structure and copy files
    print(f"\nCreating YOLO dataset structure at: {args.output}")
    create_dataset_structure(args.output, class_names, train_set, val_set, test_set)
    
    # Create YAML configuration
    create_yaml_config(args.output, class_names, len(train_set), len(val_set), len(test_set))
    
    # Print summary
    print_dataset_summary(args.output, train_set, val_set, test_set)
    
    return 0


if __name__ == '__main__':
    exit(main())
