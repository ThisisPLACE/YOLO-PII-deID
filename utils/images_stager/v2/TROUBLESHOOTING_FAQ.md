# Troubleshooting & FAQ Guide

## Installation & Setup

### Q: How do I install the required dependencies?

A:
```bash
# Install Pillow (required)
pip install Pillow

# Verify installation
python -c "from PIL import Image; print('Pillow installed successfully')"
```

### Q: What Python version do I need?

A: Python 3.7 or higher. Check your version:
```bash
python --version
# or
python3 --version
```

### Q: The script won't run, says "module not found"

A: You need to install Pillow:
```bash
pip install Pillow
# If you have multiple Python versions
pip3 install Pillow
```

### Q: "Permission denied" when running script

A: Make it executable:
```bash
chmod +x image_stitched_processor_enhanced.py

# Then run it
python image_stitched_processor_enhanced.py /input /output
```

---

## Input/Output Issues

### Q: Script says "Parent directory does not exist"

A: Verify your input path:
```bash
# Check if directory exists
ls /path/to/input
# or
dir C:\path\to\input

# Use absolute paths for clarity
python script.py /absolute/path/input /absolute/path/output

# On Windows
python script.py C:\path\to\input C:\path\to\output
```

### Q: "No folders with 'stitched' in path" but they exist

A: Possibilities:
```bash
# 1. Check if path is correct
python script.py /input /output --all-images --dry-run

# 2. Use --all-images to process all images
python script.py /input /output --all-images

# 3. Check case sensitivity (depends on OS)
# On Linux/Mac, "Stitched" ≠ "stitched"
# The script uses lowercase matching

# 4. Check for actual image files
ls /path/with/stitched/*.jpg
ls /path/with/stitched/*.png
```

### Q: Permission denied when writing to output directory

A:
```bash
# Create output directory with permissions
mkdir -p /output/path
chmod 755 /output/path

# Or use a different output directory
python script.py /input ~/images_processed
```

### Q: Output files are going to wrong location

A: The script preserves directory structure:
```
Input:  /input/project1/stitched/image1.jpg
Output: /output/project1/stitched/image1.jpg
                     ← same relative path preserved
```

If this isn't what you want:
```bash
# Use --all-images to ignore directory structure
# Actually, structure is always preserved, but you can:
# 1. Use flat output: specify output as /output and use --filename-prefix
# 2. Post-process with a script to flatten

# Move all files to single folder after processing
find /output -name "*.jpg" -type f -exec mv {} /output/flat/ \;
```

---

## Processing Issues

### Q: Script is running very slowly

A: Try these optimizations:
```bash
# 1. Reduce number of workers
python script.py /input /output --num-workers 2

# 2. Enable aggressive filtering to skip large files
python script.py /input /output --max-size 50MB --num-workers 4

# 3. Use faster preset
python script.py /input /output --preset fast

# 4. Reduce quality
python script.py /input /output --quality 50 --resize 50

# 5. Disable optimization
python script.py /input /output --optimize
# ^ This is already default (disabled), so enabling it makes it slower
```

### Q: Out of memory error

A:
```bash
# 1. Reduce number of workers (uses less RAM)
python script.py /input /output --num-workers 1

# 2. Filter out large files
python script.py /input /output --max-size 20MB

# 3. Process in batches with subdirectories
# Process folder1, then folder2, etc.
python script.py /input/folder1 /output/folder1
python script.py /input/folder2 /output/folder2
```

### Q: Processing got interrupted, can I resume?

A: Yes! Use resume mode:
```bash
# Original interrupted command:
python script.py /input /output --preset balanced

# Continue from where it stopped:
python script.py /input /output --preset balanced --resume

# Important: Use same settings as original run
```

### Q: Some images failed to process, how do I retry?

A:
```bash
# Check the log for which files failed
python script.py /input /output --log-file full.log

# View failures
grep "Failed\|Error" full.log

# Retry just those files by using different output directory
# Or use --resume and fix the problematic files manually
```

### Q: Processing is single-threaded, too slow

A: Your command is missing workers:
```bash
# Wrong (single-threaded)
python script.py /input /output --single-thread

# Right (multi-threaded)
python script.py /input /output

# Custom workers
python script.py /input /output --num-workers 8
```

---

## Quality & Output Issues

### Q: Output images look bad, poor quality

A:
```bash
# You're using the 'fast' preset (40% quality)
# Use better quality:
python script.py /input /output --preset balanced
# or
python script.py /input /output --quality 75 --resize 100
```

### Q: Output files are too large

A:
```bash
# Use more aggressive compression
python script.py /input /output --preset fast

# Or custom
python script.py /input /output --quality 50 --resize 50

# Or different format
python script.py /input /output --format WEBP --quality 60
```

### Q: Images are blurry after processing

A:
```bash
# Blurring happens with heavy resize
# Reduce resize percentage:
python script.py /input /output --resize 80
# or
python script.py /input /output --resize 100

# Or use better quality:
python script.py /input /output --quality 80
```

### Q: Colors look different in output

A: This is normal with JPEG compression. Try:
```bash
# 1. Use PNG for lossless
python script.py /input /output --preset lossless

# 2. Use higher quality
python script.py /input /output --quality 85

# 3. Use WebP (better color handling)
python script.py /input /output --preset webp-balanced
```

### Q: Output format didn't change

A:
```bash
# If you used --preserve-filename, format stays original
# Use this to force format change:
python script.py /input /output --format PNG

# Verify output
file /output/image1.png
# or
identify /output/image1.png
```

### Q: EXIF data not being preserved

A:
```bash
# Did you use the flag?
python script.py /input /output --preserve-exif

# Verify EXIF was in original
exiftool /input/image1.jpg
# or
python -c "from PIL import Image; print(Image.open('/input/image1.jpg').getexif())"
```

---

## Filtering Issues

### Q: Filter doesn't seem to work

A: Check syntax:
```bash
# Wrong: missing unit
python script.py /input /output --min-size 5

# Right: include unit
python script.py /input /output --min-size 5MB

# Size units: B, KB, MB, GB
python script.py /input /output --min-size 500KB --max-size 50MB
```

### Q: Nothing matches my filters

A:
```bash
# Use --dry-run to see what would match
python script.py /input /output --dry-run --min-width 1920

# Check what you have
find /input -name "*.jpg" | xargs identify | head
# This shows dimensions of files

# Adjust filters
python script.py /input /output --min-width 1000
```

### Q: Aspect ratio filter not working

A: Syntax issue:
```bash
# Wrong: space around comma
python script.py /input /output --aspect-ratio "0.5, 2.0"
# ^ This fails

# Right: no spaces
python script.py /input /output --aspect-ratio "0.5,2.0"

# Examples:
python script.py /input /output --aspect-ratio "0.95,1.05"   # Square
python script.py /input /output --aspect-ratio "1.3,2.0"     # Landscape
```

### Q: Date filtering not working

A: Date format must be YYYY-MM-DD:
```bash
# Wrong formats:
python script.py /input /output --modified-after "2024/01/15"
python script.py /input /output --modified-after "01-15-2024"

# Right format:
python script.py /input /output --modified-after 2024-01-15
python script.py /input /output --modified-before 2024-12-31

# Combine for date range:
python script.py /input /output \
  --modified-after 2024-01-01 \
  --modified-before 2024-12-31
```

### Q: --all-images still only processes stitched folders

A: It should work. Debug:
```bash
# Check what's being found
python script.py /input /output --all-images --dry-run --log-file preview.log
grep "Found\|Total" preview.log

# If still only finding stitched folders:
# Check that image files exist
find /input -name "*.jpg" -type f | head

# Check for read permissions
ls -la /input
```

---

## Naming Issues

### Q: Filenames changed (image.png became image.jpg)

A: Format conversion happened:
```bash
# To preserve original filename format, use:
python script.py /input /output --preserve-filename

# This keeps image.png as image.png even if processing as JPEG
```

### Q: Prefix/suffix not being applied

A: Check syntax:
```bash
# Make sure names don't have spaces
python script.py /input /output --filename-prefix "web_"

# Include extension in result
python script.py /input /output \
  --filename-prefix "compressed_" \
  --filename-suffix "_2024"
# Result: compressed_image_2024.jpg
```

### Q: Filenames have special characters causing issues

A: The script handles most characters, but:
```bash
# Avoid quotes, pipes, asterisks in filenames
python script.py /input /output --filename-prefix "web"
# Use simple strings

# If source files have problematic names:
# Rename them first before processing
rename 's/[^A-Za-z0-9._-]/_/g' /input/*
```

---

## Logging & Reporting Issues

### Q: Log file not being created

A:
```bash
# Make sure output directory exists and is writable
mkdir -p /output
chmod 755 /output

# Then run with logging
python script.py /input /output --log-file process.log

# Check if created
ls -la /output/process.log
```

### Q: Manifest is empty or missing data

A:
```bash
# Ensure you're using --manifest
python script.py /input /output --manifest report.json

# Check file was created
ls -la /output/report.json

# View contents
cat /output/report.json | python -m json.tool
# or
jq . /output/report.json
```

### Q: Can't read JSON manifest

A:
```bash
# The manifest is JSON format
# View it with:
cat report.json

# Pretty print:
python -c "import json; print(json.dumps(json.load(open('report.json')), indent=2))"

# Or use a JSON viewer tool
```

---

## Performance Tuning

### Q: How many workers should I use?

A:
```bash
# Get your CPU count:
python -c "import multiprocessing; print(multiprocessing.cpu_count())"

# Recommendation:
# - CPU count - 1 (default)
# - If system lags: CPU count - 2
# - If out of memory: 1 or 2
# - If idle system: CPU count

python script.py /input /output --num-workers 8
```

### Q: When should I use single-thread mode?

A:
```bash
# Debugging: single-threaded with logging
python script.py /input /output --single-thread --log-file debug.log

# Troubleshooting specific files:
python script.py /input /output --single-thread

# For system stability:
python script.py /input /output --num-workers 2
```

### Q: Dry-run is slow, how to make preview faster?

A:
```bash
# Dry-run still checks images (for filtering), so it's slow
# But you can:

# 1. Use aggressive filters to reduce work
python script.py /input /output --dry-run --min-size 10MB

# 2. Limit file types
python script.py /input /output --dry-run --file-types .jpg

# 3. Don't use dimension/aspect-ratio filters in preview (they open files)
# Just use size/date filters which are faster
```

---

## Common Workflow Issues

### Q: Want to process the same folder with different settings

A: Use multiple output directories:
```bash
# Version 1: Fast compression
python script.py /input /output_fast --preset fast

# Version 2: High quality
python script.py /input /output_high --preset high

# Version 3: WebP format
python script.py /input /output_webp --preset webp-balanced
```

### Q: Want to compare before/after

A:
```bash
# Check statistics before vs after
# Before:
du -sh /input
find /input -type f | wc -l

# After:
du -sh /output
find /output -type f | wc -l

# With manifest:
python script.py /input /output --manifest stats.json
# Check total_compression in stats.json
```

### Q: Want to only process new/updated files

A: Use resume with timestamps:
```bash
# First run
python script.py /input /output --manifest initial_run.json

# Later, more files added
# Use modified-after to only process new files
python script.py /input /output --modified-after 2024-01-15 --resume

# Or if re-running same command
python script.py /input /output --resume
```

### Q: Want detailed statistics about processed files

A:
```bash
# Generate manifest with all details
python script.py /input /output --manifest report.json

# View summary
python -c "import json; r=json.load(open('report.json')); print(f\"Total files: {r['summary']['total_files']}\nCompression: {r['summary']['total_compression']:.1f}%\")"

# Get per-file details
python -c "import json; r=json.load(open('report.json')); [print(f\"{f['original_dimensions']} -> {f['output_dimensions']}: {f['compression_ratio']}\") for f in r['files'][:5]]"
```

---

## Getting Help

### Q: Script doesn't work, how do I debug?

A: Use this process:
```bash
# 1. Check syntax
python script.py --help

# 2. Use dry-run to preview
python script.py /input /output --dry-run

# 3. Enable logging
python script.py /input /output --log-file debug.log

# 4. Use single-thread for detailed output
python script.py /input /output --single-thread --log-file debug.log

# 5. Check log for errors
cat debug.log | grep -i error
```

### Q: How do I see all available options?

A:
```bash
python image_stitched_processor_enhanced.py --help
```

### Q: Can I combine options?

A: Yes! Most options work together:
```bash
python script.py /input /output \
  --preset balanced \
  --min-size 1MB \
  --max-size 50MB \
  --min-width 1920 \
  --modified-after 2024-01-01 \
  --preserve-exif \
  --extract-exif \
  --manifest report.json \
  --log-file process.log

# All filters apply together (AND logic)
```

---

## Performance Reference

### Typical Processing Times

| Image Size | Compression | Time per Image |
|------------|-------------|-----------------|
| 5 MB | 40% → 2 MB | ~0.5 sec |
| 15 MB | 60% → 9 MB | ~1.2 sec |
| 30 MB | 80% → 24 MB | ~2.5 sec |

### Storage Savings by Preset

| Preset | Typical Compression |
|--------|-------------------|
| fast | 85-90% smaller |
| balanced | 75-85% smaller |
| high | 30-50% smaller |
| webp-balanced | 80-85% smaller |
| lossless | 10-30% smaller |

---

## Error Messages Explained

| Error | Meaning | Solution |
|-------|---------|----------|
| "Parent directory does not exist" | Input path is wrong | Check path, use absolute paths |
| "Permission denied" | Can't write to output | Check folder permissions, use different folder |
| "No images found" | No files match criteria | Use --all-images, check file extensions |
| "Failed to process" | Image is corrupt | Check image with `file` command, skip it |
| "Out of memory" | Too many workers | Reduce with --num-workers |
| "ModuleNotFoundError: PIL" | Pillow not installed | `pip install Pillow` |

---

## Tips for Best Results

1. **Always dry-run first** with new settings
2. **Use appropriate preset** for your use case
3. **Filter aggressively** to skip unwanted files
4. **Save manifests** for record-keeping
5. **Use resume mode** for large batches
6. **Check log files** for any errors
7. **Test with small batch** first
8. **Preserve EXIF** if metadata matters
9. **Use WebP** for modern web use
10. **Monitor disk space** during processing

