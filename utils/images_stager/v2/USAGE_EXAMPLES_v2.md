# Real-World Usage Examples

## Web Optimization Examples

### Example 1: Optimize Images for Website
```bash
# Convert images to modern WebP format for website
python image_stitched_processor_enhanced.py /photos /web_output \
  --preset webp-balanced \
  --min-width 800 \
  --filename-prefix "web_" \
  --manifest optimization_report.json

# Results:
# - Modern WebP format (~70% of JPEG size)
# - Only images wider than 800px (avoids thumbnails)
# - Named "web_image1.webp", etc.
# - Detailed report showing compression stats
```

### Example 2: Create Thumbnail Gallery
```bash
# Create tiny thumbnails for gallery index
python image_stitched_processor_enhanced.py /photos /thumbnails \
  --preset fast \
  --quality 40 \
  --resize 20 \
  --filename-suffix "_thumb" \
  --extract-exif

# Results:
# - Very small files (great for web)
# - 20% of original size (visual preview)
# - Metadata saved separately for reference
```

### Example 3: Responsive Images
```bash
# Create multiple versions for responsive design
# Mobile version
python image_stitched_processor_enhanced.py /source /output_mobile \
  --quality 50 \
  --resize 50 \
  --filename-suffix "_mobile"

# Tablet version
python image_stitched_processor_enhanced.py /source /output_tablet \
  --quality 65 \
  --resize 75 \
  --filename-suffix "_tablet"

# Desktop version
python image_stitched_processor_enhanced.py /source /output_desktop \
  --quality 80 \
  --resize 100 \
  --filename-suffix "_desktop"
```

---

## Archival Examples

### Example 4: Archive Compression
```bash
# Compress images for cold storage
python image_stitched_processor_enhanced.py /photos /archive \
  --preset fast \
  --optimize \
  --preserve-exif \
  --extract-exif \
  --manifest archive_manifest.json \
  --log-file archive_log.txt

# Results:
# - Maximum compression
# - EXIF preserved in images and extracted to JSON
# - Complete log of archive process
# - Manifest for inventory
```

### Example 5: Time-Based Archival
```bash
# Archive only old photos (before 2023)
python image_stitched_processor_enhanced.py /photos /archive_2022 \
  --modified-before 2023-01-01 \
  --preset fast \
  --filename-prefix "archive_" \
  --manifest archive_2022_inventory.json

# Archive photos from specific year
python image_stitched_processor_enhanced.py /photos /archive_2024 \
  --modified-after 2024-01-01 \
  --modified-before 2024-12-31 \
  --preset balanced \
  --preserve-exif
```

### Example 6: Selective Archival
```bash
# Archive only very large files (save space)
python image_stitched_processor_enhanced.py /photos /large_files_archive \
  --min-size 20MB \
  --preset fast \
  --quality 50 \
  --resize 50 \
  --manifest large_files_archive.json

# Keep smaller files unchanged, only compress large ones
python image_stitched_processor_enhanced.py /photos /compressed_archive \
  --min-size 1MB \
  --max-size 20MB \
  --preset balanced
```

---

## Selective Processing Examples

### Example 7: HD Only Processing
```bash
# Process only high-definition images
python image_stitched_processor_enhanced.py /photos /hd_processed \
  --min-width 1920 \
  --min-height 1080 \
  --preset high \
  --preserve-exif \
  --manifest hd_processing_report.json

# Results:
# - Only Full HD or better images processed
# - High quality output
# - Original metadata preserved
```

### Example 8: 4K Down-Sample
```bash
# Process only 4K images, reduce to 1080p
python image_stitched_processor_enhanced.py /photos /1080p_output \
  --min-width 3840 \
  --min-height 2160 \
  --quality 80 \
  --resize 28 \
  --filename-suffix "_1080p"

# Results:
# - Only 4K+ images selected
# - Resized to ~1080p resolution
# - Maintain good quality
```

### Example 9: Aspect Ratio Specific
```bash
# Process only landscape images
python image_stitched_processor_enhanced.py /photos /landscape_output \
  --aspect-ratio "1.3,2.0" \
  --preset balanced \
  --manifest landscape_report.json

# Process only portrait images
python image_stitched_processor_enhanced.py /photos /portrait_output \
  --aspect-ratio "0.5,0.77" \
  --preset balanced

# Process only square images
python image_stitched_processor_enhanced.py /photos /square_output \
  --aspect-ratio "0.95,1.05" \
  --preset balanced
```

### Example 10: Recently Modified Only
```bash
# Process only recent photos (last month)
python image_stitched_processor_enhanced.py /photos /recent_processed \
  --modified-after 2024-12-15 \
  --preset balanced \
  --preserve-exif \
  --manifest recent_processing.json

# Process new additions since last run
python image_stitched_processor_enhanced.py /photos /incremental_output \
  --modified-after 2024-12-01 \
  --resume \
  --manifest incremental_update.json
```

---

## Format Conversion Examples

### Example 11: Convert All to WebP
```bash
# Modern format conversion for all images
python image_stitched_processor_enhanced.py /photos /webp_output \
  --all-images \
  --format WEBP \
  --quality 75 \
  --resize 100 \
  --preserve-filename \
  --manifest webp_conversion.json

# Results:
# - All images converted to WebP
# - Original filenames preserved (except extension)
# - Same resolution as original
# - Detailed conversion report
```

### Example 12: Convert to Lossless PNG
```bash
# Lossless archival format
python image_stitched_processor_enhanced.py /source /png_archive \
  --preset lossless \
  --preserve-exif \
  --extract-exif \
  --manifest png_archive_manifest.json

# Results:
# - PNG format (no quality loss)
# - EXIF preserved and extracted
# - Larger files but perfect quality
```

### Example 13: Multi-Format Export
```bash
# Create multiple format versions

# For web (WebP)
python script.py /input /output_webp --format WEBP --quality 75

# For archival (PNG)
python script.py /input /output_png --format PNG --quality 95

# For compatibility (JPEG)
python script.py /input /output_jpg --format JPEG --quality 80
```

---

## Metadata Management Examples

### Example 14: Preserve All Metadata
```bash
# Process images while preserving all EXIF data
python image_stitched_processor_enhanced.py /photos /metadata_preserved \
  --preset balanced \
  --preserve-exif \
  --extract-exif \
  --manifest metadata_report.json

# Results:
# - EXIF data kept in images
# - EXIF also saved as JSON in exif_data/ folder
# - Can read metadata from JSON later
```

### Example 15: Extract and Analyze Metadata
```bash
# Extract metadata from all images for analysis
python image_stitched_processor_enhanced.py /photos /output \
  --preset balanced \
  --extract-exif \
  --manifest analysis.json

# Now analyze metadata
# exif_data/image1.json contains:
# - Camera model
# - GPS coordinates
# - Shooting date/time
# - Lens info
# - ISO, aperture, shutter speed
```

### Example 16: Photography Archive with Metadata
```bash
# Professional photography archival
python image_stitched_processor_enhanced.py /photos /archive \
  --preset high \
  --preserve-exif \
  --extract-exif \
  --all-images \
  --manifest photography_archive.json \
  --log-file archive_process.log

# Results:
# - High quality archive
# - All metadata preserved and extracted
# - Complete processing log
# - Inventory manifest
```

---

## Batch Processing Examples

### Example 17: Process Multiple Folders
```bash
# Process each project folder separately
for folder in /projects/*/; do
  project=$(basename "$folder")
  python image_stitched_processor_enhanced.py "$folder" "/output/$project" \
    --preset balanced \
    --manifest "/output/$project/report.json"
done
```

### Example 18: Incremental Processing
```bash
# First time: process everything
python image_stitched_processor_enhanced.py /photos /output \
  --preset balanced \
  --manifest initial_run.json

# Later: add new photos, process only new ones
python image_stitched_processor_enhanced.py /photos /output \
  --preset balanced \
  --resume \
  --modified-after 2024-12-01 \
  --manifest incremental_update.json
```

### Example 19: Staged Processing
```bash
# Stage 1: Preview what would be done
python image_stitched_processor_enhanced.py /photos /output \
  --preset balanced \
  --min-size 1MB \
  --dry-run \
  --manifest preview.json \
  --log-file preview.log

# Review preview.json to verify settings

# Stage 2: Actually process
python image_stitched_processor_enhanced.py /photos /output \
  --preset balanced \
  --min-size 1MB \
  --manifest full_run.json \
  --log-file full_run.log

# Stage 3: Verify results
python image_stitched_processor_enhanced.py /photos /output \
  --resume \
  --manifest final_verification.json
```

---

## Performance Examples

### Example 20: Fast Processing for Large Batches
```bash
# Optimize for speed with large image batch
python image_stitched_processor_enhanced.py /photos /output \
  --preset fast \
  --num-workers 8 \
  --max-size 20MB \
  --manifest fast_report.json

# Results:
# - Maximum compression (85%+ smaller)
# - 8 parallel workers for speed
# - Skips very large files to save time
```

### Example 21: Memory-Conscious Processing
```bash
# Process with low memory usage
python image_stitched_processor_enhanced.py /photos /output \
  --preset balanced \
  --num-workers 2 \
  --max-size 50MB

# Results:
# - Only 2 workers (low memory usage)
# - Skips very large files
# - Slower but uses less RAM
```

### Example 22: Debug Problem Images
```bash
# Find and fix problematic images
python image_stitched_processor_enhanced.py /photos /output \
  --single-thread \
  --log-file debug.log

# Check log for errors
grep -i "error\|failed" debug.log

# See detailed output for each file
tail -100 debug.log
```

---

## Naming Examples

### Example 23: Organize with Prefix/Suffix
```bash
# Add meaningful prefixes/suffixes
python image_stitched_processor_enhanced.py /photos /output \
  --filename-prefix "2024_" \
  --filename-suffix "_web" \
  --preset balanced

# Results:
# - image.jpg → 2024_image_web.jpg
# - Easy to identify by date and purpose
```

### Example 24: Preserve Original Naming
```bash
# Keep exact original names (except extension changes)
python image_stitched_processor_enhanced.py /photos /output \
  --preserve-filename \
  --format WEBP

# Results:
# - image.jpg → image.webp (only extension changes)
# - image.png → image.webp (same)
# - Filename structure preserved
```

### Example 25: Custom Naming Scheme
```bash
# For multiple output versions with clear naming
python script.py /input /output \
  --filename-prefix "optimized_" --filename-suffix "_final"
  --preset balanced

# Results:
# image.jpg → optimized_image_final.jpg
```

---

## Testing and Development Examples

### Example 26: A/B Testing Different Settings
```bash
# Test 1: Current settings
python script.py /photos /test1 --preset balanced --dry-run

# Test 2: Alternative settings
python script.py /photos /test2 --preset high --dry-run --min-width 1920

# Test 3: Aggressive compression
python script.py /photos /test3 --preset fast --dry-run

# Review results, then choose best:
python script.py /photos /output --preset balanced
```

### Example 27: Filter Tuning
```bash
# Test different filter combinations
python script.py /photos /output \
  --dry-run \
  --min-size 5MB \
  --max-size 50MB \
  --min-width 1920 \
  --manifest filter_test.json

# Check how many files match
cat filter_test.json | grep total_files

# Adjust and retry
```

### Example 28: Output Quality Validation
```bash
# Process with logging for quality verification
python script.py /photos /output \
  --preset balanced \
  --manifest detailed_report.json \
  --log-file process.log

# Analyze results
python -c "
import json
report = json.load(open('detailed_report.json'))
print(f\"Total Compression: {report['summary']['total_compression']:.1f}%\")
print(f\"Files Processed: {report['summary']['total_files']}\")
print(f\"Output Size: {report['summary']['total_output_size']}\")
"
```

---

## Professional Use Examples

### Example 29: Photography Portfolio
```bash
# Professional portfolio with high quality
python image_stitched_processor_enhanced.py /photos /portfolio \
  --preset high \
  --preserve-exif \
  --all-images \
  --filename-prefix "portfolio_" \
  --manifest portfolio_inventory.json

# Results:
# - High quality output (80% quality, full resolution)
# - Metadata preserved (important for photographers)
# - Professional naming
# - Complete inventory
```

### Example 30: Real Estate Image Processing
```bash
# Process property photos for listing
python image_stitched_processor_enhanced.py /properties /listing_images \
  --min-width 1024 \
  --quality 75 \
  --resize 100 \
  --filename-prefix "property_" \
  --manifest listing_inventory.json

# Results:
# - Professional quality output
# - Only clear, large images
# - Optimized for web display
# - Organized for multiple listings
```

### Example 31: E-commerce Product Photography
```bash
# Process product images for online store
python image_stitched_processor_enhanced.py /products /ecommerce \
  --preset webp-balanced \
  --min-width 800 \
  --all-images \
  --filename-suffix "_product" \
  --manifest product_catalog.json

# Then create versions for different uses:
# - Thumbnails: separate run with --resize 20
# - Detail view: separate run with --quality 85
# - Gallery: preset webp-balanced
```

---

## Recovery Examples

### Example 32: Resume After Interruption
```bash
# Original command was interrupted
python script.py /photos /output --preset balanced

# Check what's been done
ls /output | wc -l

# Continue from where it stopped
python script.py /photos /output --preset balanced --resume

# Verify completion
python script.py /photos /output --manifest completion_check.json
```

### Example 33: Fix Failed Processing
```bash
# After a failed run, check the log
python script.py /photos /output --preset balanced --log-file error.log 2>&1

# View errors
grep -i "error\|failed" error.log

# Fix issues (maybe: increase memory, use single-thread, exclude problem files)
python script.py /photos /output \
  --preset balanced \
  --single-thread \
  --resume \
  --num-workers 1
```

### Example 34: Verify Processing Completed
```bash
# Generate completion report
python script.py /photos /output \
  --manifest completion_report.json

# Check if all files were processed
python -c "
import json, os
report = json.load(open('completion_report.json'))
found = len(report.get('files', []))
print(f'Files in report: {found}')
print(f'Files in output directory: {sum(1 for root, dirs, files in os.walk(\"/output\") for f in files)}')
"
```

---

## Storage and Backup Examples

### Example 35: Create Backup with Compression
```bash
# Create compressed backup of all photos
python image_stitched_processor_enhanced.py /photos /backup \
  --preset fast \
  --quality 50 \
  --resize 50 \
  --optimize \
  --manifest backup_manifest.json \
  --log-file backup_log.txt

# Check size savings
du -sh /photos /backup
```

### Example 36: Cloud Storage Optimization
```bash
# Optimize images before uploading to cloud
python image_stitched_processor_enhanced.py /photos /cloud_optimized \
  --preset webp-balanced \
  --max-size 20MB \
  --filename-prefix "cloud_" \
  --manifest cloud_optimization_report.json

# Then sync to cloud:
# aws s3 sync /cloud_optimized s3://bucket/photos/
# or rclone sync /cloud_optimized remote:photos/
```

---

## Summary Table

| Use Case | Preset | Key Options |
|----------|--------|------------|
| Web optimization | webp-balanced | --min-width 800 |
| Thumbnail gallery | fast | --resize 20 |
| Archive | fast | --preserve-exif --extract-exif |
| HD processing | high | --min-width 1920 |
| 4K down-sample | balanced | --min-width 3840 --resize 28 |
| Portfolio | high | --all-images --preserve-exif |
| E-commerce | webp-balanced | --min-width 800 |
| Backup | fast | --optimize |
| Cloud upload | webp-balanced | --max-size 20MB |
| Format conversion | (choose format) | --format WEBP |

Each example can be adapted to your specific needs by combining the available options!

