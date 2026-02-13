# JPG Export with Post-Processing

## Overview

The SAHI detection script can export visualization images in JPG format by converting the PNG files that SAHI exports. This provides a flexible way to reduce file sizes while maintaining quality control.

## Configuration Options

### Basic Settings

```yaml
output:
  export_visuals: true           # Enable/disable image export
  visual_format: "jpg"           # 'png' or 'jpg'
  jpg_quality: 95                # 1-100 (see quality guide below)
```

### Default Configuration

- **Format**: JPG (recommended for most use cases)
- **Quality**: 95 (high quality with reasonable file size)
- **File Size**: ~30-50% of PNG size

---

## Quality Settings Guide

### Understanding JPG Quality

JPG uses lossy compression, meaning higher quality values preserve more image details but create larger files.

### Quality Levels and Recommendations

| Quality | Visual Quality | File Size | Use Case |
|---------|-----------------|-----------|----------|
| **50** | Low | Very small (5-10% of PNG) | Quick previews only |
| **60** | Fair | Small (10-20% of PNG) | Draft reviews |
| **70** | Good | Small-Medium (20-30% of PNG) | Most use cases ✓ |
| **80** | Very Good | Medium (30-40% of PNG) | High quality needed |
| **85** | Excellent | Medium (40-50% of PNG) | Production use ✓ |
| **90** | Excellent | Medium-Large (50-60% of PNG) | Critical analysis |
| **95** | Near-lossless | Large (60-80% of PNG) | Maximum quality |
| **100** | Maximum | Largest (can be > PNG) | Archive/Reference |

✓ = Recommended settings

---

## Configuration Examples

### Example 1: Default (Balanced Quality/Size)

```yaml
output:
  visual_format: "jpg"
  jpg_quality: 85
```

**Best for:** Most production scenarios
- Good visual quality
- Reasonable file sizes
- Fast processing
- Standard for web and reports

### Example 2: Storage Optimization

```yaml
output:
  visual_format: "jpg"
  jpg_quality: 70
```

**Best for:** Large batch processing, long-term storage
- Significantly smaller files (60-70% reduction)
- Still good visual quality
- Minimal impact on object detection visibility
- Example: 1000 images save ~1.5-2 GB

**Processing 1000 images example:**
- PNG: ~500 MB total
- JPG at quality 70: ~150-200 MB total
- JPG at quality 85: ~250-300 MB total

### Example 3: Maximum Quality

```yaml
output:
  visual_format: "jpg"
  jpg_quality: 95
```

**Best for:** Detailed analysis, documentation, archives
- Near-lossless quality
- Slight file size reduction vs PNG
- Good for critical applications
- Used for regulatory/compliance documentation

### Example 4: Quick Preview/Testing

```yaml
output:
  visual_format: "jpg"
  jpg_quality: 55
```

**Best for:** Initial testing, configuration tuning
- Very small files
- Fast processing
- Good enough for preview
- When you just need to verify detections work

### Example 5: PNG Format (No Compression Loss)

```yaml
output:
  visual_format: "png"
```

**Best for:** When lossless quality is critical
- Perfect image quality
- Larger file sizes
- Slower to save
- Supports transparency (though detection visuals don't use it)

---

## Performance Impact

### File Size Comparison

For a typical 4K detection image (4096×2160):

| Format | Quality | File Size | Compression |
|--------|---------|-----------|-------------|
| PNG | - | 8-15 MB | Lossless |
| JPG | 50 | 500 KB | 94-97% smaller |
| JPG | 70 | 1.2 MB | 85-92% smaller |
| JPG | 85 | 2.5 MB | 75-85% smaller |
| JPG | 95 | 5 MB | 35-55% smaller |

### Processing Speed

JPG conversion adds minimal overhead:
- **Conversion time**: ~0.5-1 second per image
- **Total time per image**: ~3-5 seconds (with 32 slices)
- **Negligible impact** on overall processing

### Storage Requirements

**Processing 500 images at different settings:**

| Quality | Total Size | Disk Space Saved |
|---------|-----------|-----------------|
| PNG | ~2.5 GB | Baseline |
| JPG 50 | ~50 MB | 2.45 GB (98%) |
| JPG 70 | ~300 MB | 2.2 GB (88%) |
| JPG 85 | ~1 GB | 1.5 GB (60%) |
| JPG 95 | ~1.8 GB | 700 MB (28%) |

---

## Technical Details

### How It Works

**Process:**
1. SAHI exports detection visualization as PNG (standard behavior)
2. If `visual_format: "jpg"` is configured, PNG is converted to JPG using PIL/Pillow
3. JPG file is saved with the specified quality level
4. PNG file is deleted to save disk space
5. RGBA images are converted to RGB with white background (JPG requirement)

### Conversion Pipeline

```
Detection Results
      ↓
SAHI export_visuals()
      ↓
PNG file (_prediction.png)
      ↓
PIL conversion to JPG
      ↓
JPG file (_prediction.jpg)
      ↓
PNG deleted
```

### Requirements

- **PIL/Pillow**: For PNG→JPG conversion
- Install with: `pip install Pillow`

---

## Recommended Configurations

### For Different Use Cases

**Use Case: Website/Online Sharing**
```yaml
output:
  visual_format: "jpg"
  jpg_quality: 75
```
- Small file sizes for fast loading
- Sufficient quality for web viewing
- Typical: 200-400 KB per image

**Use Case: Quality Assurance/Review**
```yaml
output:
  visual_format: "jpg"
  jpg_quality: 85
```
- High quality for detailed review
- Reasonable file sizes
- Typical: 800 KB - 1.5 MB per image

**Use Case: Data Archival**
```yaml
output:
  visual_format: "jpg"
  jpg_quality: 90
```
- Near-original quality
- Professional/regulatory compliance
- Typical: 1.5-3 MB per image

**Use Case: High-Volume Batch Processing**
```yaml
output:
  visual_format: "jpg"
  jpg_quality: 65
```
- Minimize storage requirements
- Still acceptable quality
- Typical: 300-600 KB per image
- Save ~85% disk space vs PNG

**Use Case: Testing/Development**
```yaml
output:
  visual_format: "jpg"
  jpg_quality: 50
```
- Fastest processing
- Smallest file size
- Not for production
- Typical: 100-250 KB per image

---

## Troubleshooting

### Issue: "PIL/Pillow not installed"

**Error:**
```
[ERROR] PIL/Pillow not installed. Install with: pip install Pillow
```

**Solution:**
```bash
pip install Pillow
```

### Issue: Still Exporting PNG Files

**Cause:** `visual_format` is not set to "jpg"

**Solution:** Check config.yaml:
```yaml
output:
  visual_format: "jpg"    # Must be exactly "jpg"
```

### Issue: Invalid JPG Quality Warning

**Problem:** 
```
WARNING: Invalid JPG quality 150, using default 95
```

**Solution:** Set quality between 1-100:
```yaml
jpg_quality: 85  # Valid range: 1-100
```

### Issue: JPG Conversion is Slow

**Cause:** Processing large/high-resolution images

**Solution:** 
- Reduce `jpg_quality` for faster conversion
- Test with smaller batch first (`max_images: 10`)

## Requirements

- **Pillow/PIL**: `pip install Pillow`

Once installed, JPG conversion will work automatically when configured.

---

## Comparison: JPG vs PNG

### JPG Advantages
✓ Much smaller file sizes (70-95% reduction)
✓ Faster to save and load
✓ Universal compatibility
✓ Good for web/sharing
✓ Reasonable quality-size trade-off

### JPG Disadvantages
✗ Lossy compression (slight quality loss)
✗ No transparency support
✗ Less suitable for archival (PNG better)

### PNG Advantages
✓ Lossless compression (no quality loss)
✓ Supports transparency
✓ Better for long-term archival
✓ Professional documents

### PNG Disadvantages
✗ Much larger file sizes
✗ Slower to save/load
✗ More storage required

---

## Best Practices

1. **Choose appropriate quality for your use case**
   - Web: 70-75
   - Reports: 85
   - Archive: 90-95

2. **Test with small batch first**
   - Process 10 images
   - Review quality
   - Adjust if needed

3. **Monitor storage space**
   - Large batches can consume significant disk space
   - Plan accordingly for thousands of images

4. **Consider downstream use**
   - If images will be cropped/processed: use higher quality
   - If just for review: lower quality is fine

5. **Document your settings**
   - Save your config.yaml as reference
   - Document quality decisions for compliance

---

## Example Workflow

### Step 1: Configure

```yaml
output:
  export_visuals: true
  visual_format: "jpg"
  jpg_quality: 80
```

### Step 2: Test with Sample

```python
# Process just 5 images to test quality
image_processing:
  max_images: 5
```

### Step 3: Review Results

```bash
# Check file sizes and visual quality
dir output_dir /s
# Open a few images in image viewer
# Verify bounding boxes are clear
```

### Step 4: Full Processing

```python
# Remove max_images limit
image_processing:
  max_images: null
```

```bash
python sahi_detection.py
```

### Step 5: Verify and Archive

```bash
# Check total output size
du -sh output_dir/

# Archive if needed
zip -r results.zip output_dir/
```

---

## Questions?

Refer to the main README.md for overall pipeline documentation.

For YOLO format export, see YOLO_FORMAT_GUIDE.md
