#!/bin/bash
# Example usage scenarios for create_yolo_dataset.py

# Example 1: Create a basic dataset with 200 images (default)
# python create_yolo_dataset.py \
#     -i ~/datasets/raw_images \
#     -o ~/datasets/yolo_dataset \
#     --seed 42

# Example 2: Create a larger dataset with 500 images
# python create_yolo_dataset.py \
#     -i ~/datasets/raw_images \
#     -o ~/datasets/yolo_dataset_500 \
#     -n 500 \
#     --seed 42

# Example 3: Custom split ratio (80% train, 15% val, 5% test)
# python create_yolo_dataset.py \
#     -i ~/datasets/raw_images \
#     -o ~/datasets/yolo_dataset_custom \
#     -n 1000 \
#     --train 0.8 \
#     --val 0.15 \
#     --test 0.05

# Example 4: Minimal test split (good for large datasets)
# python create_yolo_dataset.py \
#     -i ~/datasets/raw_images \
#     -o ~/datasets/yolo_dataset_large \
#     -n 5000 \
#     --train 0.85 \
#     --val 0.10 \
#     --test 0.05

# Example 5: Create a small dataset for prototyping
# python create_yolo_dataset.py \
#     -i ~/datasets/raw_images \
#     -o ~/datasets/yolo_dataset_small \
#     -n 50 \
#     --train 0.7 \
#     --val 0.2 \
#     --test 0.1 \
#     --seed 123

# ============================================================================
# After creating the dataset, you can train with YOLOv8:
# ============================================================================
