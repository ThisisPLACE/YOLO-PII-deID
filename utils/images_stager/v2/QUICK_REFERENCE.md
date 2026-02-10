# Image Processor - Quick Reference

## One-Liners for Common Tasks

### Basic Operations
```bash
# Fast compression (default settings)
python script.py /input /output

# Use quality preset
python script.py /input /output --preset balanced

# Process all images (not just stitched)
python script.py /input /output --all-images

# Preview what would happen
python script.py /input /output --dry-run

# Resume interrupted processing
python script.py /input /output --resume
```

---

## Quality Presets Quick List

```bash
--preset fast              # 40% quality, 50% res → smallest files
--preset balanced          # 60% quality, 75% res → RECOMMENDED
--preset high              # 80% quality, 100% res → best quality
--preset lossless          # 95% quality, PNG → no loss
--preset webp-fast         # 50% quality, 50% res → modern format
--preset webp-balanced     # 70% quality, 75% res → modern balanced
```

---

## Custom Quality & Resolution

```bash
--quality 60               # Quality 1-100 (default 60)
--resize 75                # Resize to 75% of original (default 75)
--format JPEG/PNG/WEBP/BMP # Output format (default JPEG)
--optimize                 # Slower but smaller files
```

---

## File Processing Modes

```bash
--all-images              # All images (default: only "stitched" folders)
--recursive               # Search all subdirs (default: true)
--no-recursive            # Only top level
--resume                  # Skip existing files
--dry-run                 # Preview without saving
--single-thread           # Debug mode, one file at a time
```

---

## Filtering Options

### Size Filtering
```bash
--min-size 1MB            # At least 1MB
--max-size 50MB           # At most 50MB
--min-size 5MB --max-size 100MB    # Between 5-100MB
```

### Image Dimensions
```bash
--min-width 1920          # At least 1920px wide
--min-height 1080         # At least 1080px tall
--max-width 4000          # At most 4000px wide
--aspect-ratio 0.5,2.0    # Aspect ratio range
```

### Date Range
```bash
--modified-after 2024-01-01         # After this date
--modified-before 2024-12-31        # Before this date
--modified-after 2024-01-01 --modified-before 2024-12-31  # During 2024
```

### File Types
```bash
--file-types .jpg .png    # Only these types
```

---

## Naming & Output

```bash
--preserve-filename       # Keep original filename
--filename-prefix "web_"  # Add prefix
--filename-suffix "_opt"  # Add suffix
--exclude-dirs temp backup    # Skip these directories
```

---

## EXIF & Metadata

```bash
--preserve-exif           # Keep EXIF in output
--extract-exif            # Save EXIF to JSON files
--preserve-exif --extract-exif   # Both
```

---

## Performance & Logging

```bash
--num-workers 4           # Use 4 parallel workers
--log-file process.log    # Save log to file
--manifest report.json    # Generate detailed report
```

---

## Complete Examples

### Web Optimization
```bash
python script.py /input /output \
  --preset webp-balanced \
  --min-width 800 \
  --filename-prefix "web_"
```

### Archive Compression
```bash
python script.py /input /output \
  --preset fast \
  --quality 40 --resize 50 \
  --optimize
```

### Selective Processing
```bash
python script.py /input /output \
  --min-size 5MB --max-size 100MB \
  --min-width 1920 \
  --modified-after 2024-01-01
```

### EXIF Preservation
```bash
python script.py /input /output \
  --preset high \
  --preserve-exif \
  --extract-exif \
  --manifest report.json
```

### High Quality, Original Size
```bash
python script.py /input /output \
  --preset high
```

### Convert All to WebP
```bash
python script.py /input /output \
  --all-images \
  --format WEBP \
  --quality 75
```

### Debug Mode
```bash
python script.py /input /output \
  --dry-run \
  --single-thread \
  --log-file debug.log
```

---

## Helpful Combinations

### "Safe" Mode - Verify First
```bash
# 1. Preview
python script.py /input /output --dry-run

# 2. Actually run
python script.py /input /output
```

### Resume After Interruption
```bash
# Was interrupted, continue:
python script.py /input /output --resume
```

### Find Large Old Files and Compress
```bash
python script.py /input /output \
  --min-size 10MB \
  --modified-before 2023-12-31 \
  --preset fast
```

### Only Process Recent HD Images
```bash
python script.py /input /output \
  --min-width 1920 \
  --modified-after 2024-01-01 \
  --preset balanced
```

### Generate Detailed Report
```bash
python script.py /input /output \
  --manifest report.json \
  --log-file process.log
```

---

## Size Units

```
--min-size 500B    # 500 bytes
--min-size 500KB   # 500 kilobytes
--min-size 5MB     # 5 megabytes
--min-size 2GB     # 2 gigabytes
```

---

## Aspect Ratio Examples

```bash
--aspect-ratio "0.95,1.05"    # Nearly square (1:1)
--aspect-ratio "1.3,2.0"      # Landscape/wide
--aspect-ratio "0.5,0.77"     # Portrait/tall
--aspect-ratio "1.77,1.85"    # 16:9 to 16:8
```

---

## File Format Choice

| Goal | Command |
|------|---------|
| **Best compression** | `--preset fast` |
| **Balanced** | `--preset balanced` |
| **High quality** | `--preset high` |
| **No quality loss** | `--preset lossless` |
| **Modern web** | `--preset webp-balanced` |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | `--num-workers 2` (fewer workers) |
| Need to debug | `--single-thread --log-file debug.log` |
| Want to verify settings | `--dry-run` (preview) |
| Was interrupted | `--resume` (continue) |
| Want detailed report | `--manifest report.json` |

---

## Pro Tips

1. **Always dry-run first** with complex filters: `--dry-run`
2. **Use resume mode** for large batches: `--resume`
3. **Filter aggressively** to skip unwanted files: `--min-size`, `--max-width`
4. **Check the manifest** for statistics: `--manifest report.json`
5. **Single-thread for debugging**: `--single-thread`
6. **Use presets** - they're optimized: `--preset balanced`
7. **Combine filters** - they all apply together (AND logic)

---

## Full Help

```bash
python script.py --help
```

This shows all available options with descriptions.

