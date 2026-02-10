# Image Stitched Processor - Enhanced Edition
## Complete Feature Documentation

---

## Overview

The enhanced image processor is a powerful, flexible tool for batch processing images with extensive configuration options. It supports quality presets, advanced filtering, EXIF handling, resume capability, dry-run testing, and comprehensive logging.

---

## Quick Start

### Basic Usage
```bash
python image_stitched_processor_enhanced.py /input/path /output/path
```

### Using Quality Presets
```bash
python image_stitched_processor_enhanced.py /input/path /output/path --preset balanced
```

### Process All Images (Not Just "Stitched")
```bash
python image_stitched_processor_enhanced.py /input/path /output/path --all-images
```

---

## Quality Presets

Six built-in presets for common use cases:

| Preset | Quality | Resolution | Format | Use Case |
|--------|---------|------------|--------|----------|
| **fast** | 40% | 50% | JPEG | Maximum compression for thumbnails |
| **balanced** | 60% | 75% | JPEG | Good quality/size balance (recommended) |
| **high** | 80% | 100% | JPEG | High quality, original resolution |
| **lossless** | 95% | 100% | PNG | No quality loss (larger files) |
| **webp-fast** | 50% | 50% | WEBP | Modern format, fast compression |
| **webp-balanced** | 70% | 75% | WEBP | Modern format, good balance |

### Examples
```bash
# Fast compression - for web thumbnails
python script.py /input /output --preset fast

# Balanced quality/size - recommended for most use
python script.py /input /output --preset balanced

# High quality with original resolution
python script.py /input /output --preset high

# Lossless PNG compression
python script.py /input /output --preset lossless

# Modern WebP format
python script.py /input /output --preset webp-balanced
```

---

## Custom Quality & Format Options

### Quality and Resolution
```bash
# Custom quality (1-100) and resize percentage
python script.py /input /output --quality 75 --resize 80

# Keep original resolution, just compress
python script.py /input /output --quality 70 --resize 100

# Heavy compression for thumbnails
python script.py /input /output --quality 40 --resize 50
```

### Output Format
```bash
# JPEG (default)
python script.py /input /output --format JPEG

# PNG (lossless)
python script.py /input /output --format PNG

# WebP (modern, efficient)
python script.py /input /output --format WEBP --quality 75

# BMP (uncompressed, large files)
python script.py /input /output --format BMP
```

### Optimization
```bash
# Enable PIL optimization (slower but potentially smaller files)
python script.py /input /output --optimize
```

---

## File Naming Options

### Preserve Original Filenames
```bash
# Keep original filenames even when changing format
python script.py /input /output --preserve-filename
```

### Add Prefix/Suffix
```bash
# Add prefix to all output files
python script.py /input /output --filename-prefix "compressed_"

# Add suffix
python script.py /input /output --filename-suffix "_web"

# Both
python script.py /input /output --filename-prefix "web_" --filename-suffix "_optimized"
# Result: web_image_optimized.jpg
```

---

## Processing Mode Options

### Recursive/Non-Recursive Search
```bash
# Search recursively (default)
python script.py /input /output --recursive

# Only top-level directory
python script.py /input /output --no-recursive
```

### Process Specific Folders
```bash
# Only process folders with "stitched" in path (default)
python script.py /input /output

# Process ALL images in directory
python script.py /input /output --all-images

# Exclude specific directories
python script.py /input /output --exclude-dirs temp cache .git
```

### Resume Interrupted Processing
```bash
# Skip files that already exist in output
python script.py /input /output --resume

# Useful after interruption - continue where you left off
```

### Dry-Run Mode (Preview)
```bash
# See what would be processed without saving anything
python script.py /input /output --dry-run

# Combine with other options to preview filtering
python script.py /input /output --dry-run --min-size 5MB --max-size 50MB
```

---

## Advanced Filtering

### File Size Filtering
```bash
# Only process files between 1MB and 50MB
python script.py /input /output --min-size 1MB --max-size 50MB

# Only large files
python script.py /input /output --min-size 10MB

# Only small files
python script.py /input /output --max-size 2MB

# Size units: B, KB, MB, GB
```

### Image Dimension Filtering
```bash
# Only HD images and larger
python script.py /input /output --min-width 1920 --min-height 1080

# Only images under 4K
python script.py /input /output --max-width 3840 --max-height 2160

# Square images only (1:1 aspect ratio)
python script.py /input /output --aspect-ratio "0.95,1.05"

# Landscape images (16:9 range: 1.77:1)
python script.py /input /output --aspect-ratio "1.5,2.0"

# Portrait images
python script.py /input /output --aspect-ratio "0.5,0.7"
```

### Date-Based Filtering
```bash
# Only images modified in 2024
python script.py /input /output --modified-after 2024-01-01 --modified-before 2024-12-31

# Recent images only
python script.py /input /output --modified-after 2024-01-01

# Images before specific date
python script.py /input /output --modified-before 2023-12-31
```

### File Type Filtering
```bash
# Only JPG and PNG
python script.py /input /output --file-types .jpg .png

# Only WebP
python script.py /input /output --file-types .webp
```

### Combined Filters (AND logic)
```bash
# Complex filtering: HD images, 5-50MB, recent, landscape format
python script.py /input /output \
  --min-width 1920 \
  --min-size 5MB \
  --max-size 50MB \
  --modified-after 2024-01-01 \
  --aspect-ratio "1.5,2.0"
```

---

## EXIF and Metadata Handling

### Preserve EXIF Data
```bash
# Keep EXIF data in output images
python script.py /input /output --preserve-exif

# Useful for maintaining camera metadata, GPS info, etc.
```

### Extract EXIF Data
```bash
# Save EXIF data as separate JSON files
python script.py /input /output --extract-exif

# Creates: output_dir/exif_data/image_name.json
```

### Both Preserve and Extract
```bash
# Keep EXIF in images AND save as JSON
python script.py /input /output --preserve-exif --extract-exif
```

---

## Performance and Debugging

### Multi-Worker Processing (Default)
```bash
# Use automatic worker count (CPU count - 1)
python script.py /input /output

# Specify exact number of workers
python script.py /input /output --num-workers 4

# Useful for controlling system load
```

### Single-Threaded Mode
```bash
# Debug mode - see detailed logging for each image
python script.py /input /output --single-thread

# Shows:
# - Processing path
# - Original dimensions
# - Any errors with full context
```

---

## Logging and Reporting

### Log to File
```bash
# Save processing log
python script.py /input /output --log-file processing.log

# Creates timestamped log with all operations
```

### Generate Processing Manifest
```bash
# Create detailed JSON report of all processed files
python script.py /input /output --manifest report.json

# Includes:
# - Configuration used
# - Summary statistics
# - Per-file compression details
# - Processing times
# - Original vs output dimensions
```

### View Manifest Contents
The generated manifest contains:
```json
{
  "timestamp": "2024-01-15T10:30:45.123456",
  "processing_duration": "0:03:45",
  "summary": {
    "total_files": 150,
    "total_input_size": "2.5GB",
    "total_output_size": "450.3MB",
    "total_compression": "82.0%",
    "average_processing_time_per_file": "1.5 seconds"
  },
  "files": [
    {
      "compression_ratio": "85.2%",
      "input_size": "15.3MB",
      "output_size": "2.3MB",
      "original_dimensions": [4000, 3000],
      "output_dimensions": [3000, 2250]
    }
  ]
}
```

---

## Complex Examples

### Example 1: Web Optimization
```bash
# Optimize images for web delivery
python script.py /input /output \
  --preset webp-balanced \
  --min-width 800 \
  --filename-prefix "web_" \
  --manifest web_optimization_report.json
```

### Example 2: Archive Compression
```bash
# Maximum compression for archival
python script.py /input /output \
  --preset fast \
  --quality 40 \
  --resize 50 \
  --optimize \
  --log-file archive_log.txt
```

### Example 3: Selective Processing
```bash
# Process only recent, large, HD images
python script.py /input /output \
  --min-size 5MB \
  --max-size 100MB \
  --min-width 1920 \
  --modified-after 2024-01-01 \
  --preserve-exif \
  --manifest selective_report.json
```

### Example 4: Resume with Verification
```bash
# First run with dry-run to verify
python script.py /input /output --dry-run --preset balanced

# Actually process
python script.py /input /output --preset balanced

# If interrupted, continue
python script.py /input /output --preset balanced --resume
```

### Example 5: EXIF Preservation for Archives
```bash
# Process while preserving all metadata
python script.py /input /output \
  --preset high \
  --preserve-exif \
  --extract-exif \
  --manifest metadata_report.json
```

### Example 6: Batch Convert All Images
```bash
# Convert all images to WebP
python script.py /input /output \
  --all-images \
  --format WEBP \
  --quality 75 \
  --resize 100
```

---

## Advanced Techniques

### Two-Stage Processing
```bash
# Stage 1: Preview and validate
python script.py /input /temp --dry-run --preset balanced

# Stage 2: Actual processing with same settings
python script.py /input /output --preset balanced

# Stage 3: If needed, continue after interruption
python script.py /input /output --preset balanced --resume
```

### Multi-Format Output
```bash
# Process once for high quality
python script.py /input /output_high --preset high

# Process again for web (requires running twice)
python script.py /input /output_web --preset webp-balanced
```

### Filtered Subsets
```bash
# Create optimized version of recent large files
python script.py /input /output \
  --modified-after 2024-01-01 \
  --min-size 10MB \
  --preset fast \
  --filename-suffix "_fast"
```

### Aspect Ratio Based Processing
```bash
# Process only square images
python script.py /input /output_square --aspect-ratio "0.95,1.05"

# Process only landscape
python script.py /input /output_landscape --aspect-ratio "1.3,2.0"

# Process only portrait
python script.py /input /output_portrait --aspect-ratio "0.5,0.77"
```

---

## Output Structure

### Default Behavior
```
input/
├── project1/
│   └── stitched/
│       └── image1.jpg

Becomes:

output/
└── project1/
    └── stitched/
        └── image1.jpg (processed)
```

### With EXIF Extraction
```
output/
├── project1/
│   └── stitched/
│       └── image1.jpg (processed, EXIF preserved)
└── exif_data/
    └── project1/
        └── stitched/
            └── image1.json (extracted EXIF)
```

### With Manifest
```
output/
├── project1/
│   └── stitched/
│       └── image1.jpg
├── processing_report.json
└── processing.log (if --log-file specified)
```

---

## Performance Tips

1. **Use appropriate quality presets**: `balanced` is usually best
2. **Enable resume mode**: `--resume` saves time if interrupted
3. **Use dry-run first**: `--dry-run` helps verify filters work correctly
4. **Adjust worker count**: Default (CPU-1) is usually optimal
5. **Filter aggressively**: Use `--min-size`, `--max-width` to skip unwanted files
6. **Enable optimization selectively**: `--optimize` is slower but smaller
7. **Use single-thread for debugging**: `--single-thread` for troubleshooting

---

## Troubleshooting

### Memory Issues with Large Files
```bash
# Process with fewer workers to reduce memory usage
python script.py /input /output --num-workers 2
```

### Need to Debug Processing
```bash
# Single-threaded mode with detailed logging
python script.py /input /output --single-thread --log-file debug.log
```

### Want to Verify Settings First
```bash
# Dry-run shows what would be processed
python script.py /input /output --dry-run --log-file preview.log
```

### Processing Interrupted
```bash
# Resume from where it stopped
python script.py /input /output --resume
```

### Need to Try Different Settings
```bash
# Use separate output directory for each test
python script.py /input /output_test1 --preset balanced --dry-run
python script.py /input /output_test2 --preset fast --dry-run
```

---

## File Format Recommendations

| Use Case | Format | Quality | Resize |
|----------|--------|---------|--------|
| Web thumbnails | WEBP | 50% | 50% |
| Web display | WEBP | 75% | 75% |
| High quality archive | PNG | 95% | 100% |
| Balanced web | JPEG | 75% | 75% |
| Maximum compression | JPEG | 40% | 50% |
| No quality loss | PNG | 95% | 100% |
| Photography | JPEG | 85% | 100% |
| Screenshots | PNG | 95% | 100% |

---

## Command Reference

### Quality/Format Arguments
- `--preset {fast,balanced,high,lossless,webp-fast,webp-balanced}` - Use quality preset
- `--quality N` - Output quality (1-100)
- `--resize N` - Resize to N% of original
- `--format {JPEG,PNG,WEBP,BMP}` - Output format
- `--optimize` - Enable PIL optimization
- `--preserve-filename` - Keep original filename
- `--filename-prefix STR` - Add prefix to filenames
- `--filename-suffix STR` - Add suffix to filenames

### Processing Arguments
- `--all-images` - Process all images (not just stitched)
- `--recursive` / `--no-recursive` - Enable/disable recursive search
- `--exclude-dirs D1 D2...` - Exclude directories
- `--resume` - Skip existing files
- `--dry-run` - Preview without saving

### Filter Arguments
- `--min-size N` - Minimum file size (e.g., 1MB)
- `--max-size N` - Maximum file size
- `--min-width N` - Minimum image width pixels
- `--max-width N` - Maximum image width
- `--min-height N` - Minimum image height
- `--max-height N` - Maximum image height
- `--aspect-ratio MIN,MAX` - Aspect ratio range
- `--modified-after YYYY-MM-DD` - Modified after date
- `--modified-before YYYY-MM-DD` - Modified before date
- `--file-types .jpg .png...` - File types to process

### EXIF Arguments
- `--preserve-exif` - Keep EXIF in output
- `--extract-exif` - Save EXIF as JSON

### Performance Arguments
- `--num-workers N` - Number of parallel workers
- `--single-thread` - Single-threaded debug mode

### Logging Arguments
- `--log-file PATH` - Save log to file
- `--manifest PATH` - Generate JSON report

---

## Requirements

- Python 3.7+
- Pillow (PIL) library
- Standard library modules: os, sys, json, time, logging, pathlib, datetime, multiprocessing, argparse, traceback

Install requirements:
```bash
pip install Pillow
```

---

## License and Usage

This tool is designed for batch image processing. Use responsibly and ensure you have rights to process images.

For questions or issues, refer to PIL/Pillow documentation:
- https://python-pillow.org/
- https://pillow.readthedocs.io/

