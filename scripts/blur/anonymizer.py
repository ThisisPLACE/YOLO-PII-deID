"""
Blur anonymizer script - processes CSV-based detections and blurs regions in images
Preserves directory structure in output folder
Preserves EXIF metadata
"""

import sys
import os
import cv2
import csv
from pathlib import Path
from datetime import datetime
import piexif
import multiprocessing


# Global log storage
log_messages = []


def log_message(message, print_output=True):
    """Store and optionally print log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    log_messages.append(formatted_msg)
    if print_output:
        print(message)


def exif_inject(origin, new):
    """Copy EXIF data from origin image to new image"""
    try:
        exif_dict = piexif.load(origin)
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, new)
    except Exception as e:
        log_message(f"  Warning: Could not preserve EXIF - {e}")


def blur_region(img, x, y, w, h, img_path):
    """Blur a region defined by normalized coordinates"""
    try:
        # Convert normalized coordinates to pixel coordinates
        ty = int((y - h/2) * img.shape[0])
        by = int((y + h/2) * img.shape[0])
        lx = int((x - w/2) * img.shape[1])
        rx = int((x + w/2) * img.shape[1])
        
        # Clamp to image bounds
        ty = max(0, ty)
        by = min(img.shape[0], by)
        lx = max(0, lx)
        rx = min(img.shape[1], rx)
        
        # Apply blur
        blurred_part = cv2.blur(img[ty:by, lx:rx], (40, 40))
        img[ty:by, lx:rx] = blurred_part
        return img, True
    except Exception as e:
        log_message(f"  Warning: Could not blur region in {img_path} - {e}")
        return img, False


def process_detections_csv(csv_path, parent_dir, output_dir):
    """
    Process CSV file and blur all detected regions in images
    """
    
    log_message(f"Starting blur processing...")
    log_message(f"  CSV File: {csv_path}")
    log_message(f"  Parent Dir: {parent_dir if parent_dir else 'None (using CSV paths as-is)'}")
    log_message(f"  Output Dir: {output_dir}\n")
    
    # Statistics
    processed_count = 0
    failed_images = []
    warning_list = []
    detection_count = 0
    
    # Group detections by image path
    detections_by_image = {}
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row['image_path'].replace('\\', os.sep).lstrip(os.sep)
                
                # Construct full path
                if parent_dir:
                    full_img_path = os.path.join(parent_dir, img_path)
                else:
                    full_img_path = img_path
                
                if full_img_path not in detections_by_image:
                    detections_by_image[full_img_path] = []
                
                detections_by_image[full_img_path].append({
                    'class_id': int(row['class_id']),
                    # Multiprocessing worker
                    def process_image_worker(args):
                        full_img_path, detections, parent_dir, output_dir = args
                        result = {'img_path': full_img_path, 'status': '', 'blur_count': 0, 'warnings': []}
                        # Create output directory structure
                        if parent_dir:
                            rel_path = os.path.relpath(full_img_path, parent_dir)
                        else:
                            rel_path = os.path.basename(full_img_path)
                        output_path = os.path.join(output_dir, rel_path)
                        output_subdir = os.path.dirname(output_path)
                        os.makedirs(output_subdir, exist_ok=True)

                        # Skip if output file already exists
                        if os.path.exists(output_path):
                            result['status'] = 'skipped'
                            return result

                        if not os.path.exists(full_img_path):
                            result['status'] = 'not_found'
                            result['warnings'].append(f"Image not found: {full_img_path}")
                            return result

                        try:
                            img = cv2.imread(full_img_path)
                            if img is None:
                                result['status'] = 'load_fail'
                                result['warnings'].append(f"Could not load image: {full_img_path}")
                                return result
                            blur_count = 0
                            for detection in detections:
                                img, success = blur_region(
                                    img,
                                    detection['x'],
                                    detection['y'],
                                    detection['w'],
                                    detection['h'],
                                    full_img_path
                                )
                                if success:
                                    blur_count += 1
                            cv2.imwrite(output_path, img)
                            exif_inject(full_img_path, output_path)
                            result['status'] = 'processed'
                            result['blur_count'] = blur_count
                        except Exception as e:
                            result['status'] = 'error'
                            result['warnings'].append(f"Error processing {full_img_path}: {e}")
                        return result

                    # Prepare arguments for multiprocessing
                    image_args = [
                        (full_img_path, detections, parent_dir, output_dir)
                        for full_img_path, detections in detections_by_image.items()
                    ]
                    total_images = len(image_args)

                    # Use all available CPU cores
                    cpu_count = multiprocessing.cpu_count()
                    log_message(f"Using {cpu_count} CPU cores for parallel processing.")

                    with multiprocessing.Pool(cpu_count) as pool:
                        results = []
                        for i, result in enumerate(pool.imap_unordered(process_image_worker, image_args), 1):
                            percent = i / total_images
                            bar_length = 40
                            filled_length = int(bar_length * percent)
                            bar = '█' * filled_length + '-' * (bar_length - filled_length)
                            progress_msg = f"[{bar}] {int(percent*100)}% ({i}/{total_images})"
                            print(progress_msg, end='\r')
                            results.append(result)
                        print()

                    # Collect results
                    processed_count = 0
                    failed_images = []
                    warning_list = []
                    detection_count = 0
                    for result in results:
                        detection_count += result['blur_count']
                        if result['status'] == 'processed':
                            processed_count += 1
                            log_message(f"✓ Processed: {result['img_path']} ({result['blur_count']} regions blurred)")
                        elif result['status'] == 'skipped':
                            log_message(f"⏩ Skipped (already processed): {result['img_path']}")
                        else:
                            failed_images.append(result['img_path'])
                            for w in result['warnings']:
                                warning_list.append(w)
                                log_message(f"⚠ {w}")
    log_message("\n" + "="*70)
    log_message("PROCESSING SUMMARY")
    log_message("="*70)
    log_message(f"✓ Successfully processed: {processed_count} images")
    log_message(f"  Total detections blurred: {detection_count}")
    log_message(f"✗ Failed to process: {len(failed_images)} images")
    
    if warning_list:
        log_message(f"\n⚠ WARNINGS ({len(warning_list)}):")
        for warning in warning_list:
            log_message(f"  - {warning}")
    
    if not parent_dir:
        log_message(f"\n ℹ Note: No parent directory supplied. CSV paths used as-is.")
    
    log_message("="*70)
    
    # Write log file
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, "processing_log.txt")
    try:
        with open(log_file_path, 'w') as log_file:
            for msg in log_messages:
                log_file.write(msg + '\n')
        log_message(f"\n✓ Log file saved: {log_file_path}")
    except Exception as e:
        log_message(f"\n⚠ Could not write log file: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python anonymizer.py <csv_file> <output_dir> [parent_dir]")
        print("\nArguments:")
        print("  csv_file    - Path to CSV file with detections")
        print("  output_dir  - Directory to save blurred images (preserves structure)")
        print("  parent_dir  - (Optional) Parent directory to prepend to CSV image paths")
        print("\nExample:")
        print("  python anonymizer.py sample_input_file.txt output C:\\images")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_dir = sys.argv[2]
    parent_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(csv_file):
        print(f"ERROR: CSV file not found: {csv_file}")
        sys.exit(1)
    
    process_detections_csv(csv_file, parent_dir, output_dir)
