# Configuration File Guide

This guide explains how to use and customize JSON configuration files for the dual-model SAHI detection script.

## Quick Start

1. Copy one of the example config files
2. Edit the paths to match your setup
3. Run: `python sahi_dual_model.py --config your_config.json`

## Available Config Files

### config.json (Default)
- **Purpose**: General-purpose detection on all images
- **Visualization**: Disabled (faster)
- **Directory Filter**: None (processes all)
- **Use case**: Production runs on full datasets

### config_test.json
- **Purpose**: Quick testing with visual verification
- **Visualization**: 10 random samples
- **Directory Filter**: None
- **Use case**: Initial testing, parameter tuning

### config_stitched.json
- **Purpose**: Process only "stitched" directories
- **Visualization**: Disabled
- **Directory Filter**: "stitched"
- **Use case**: Processing specific subsets of data

### config_highres.json
- **Purpose**: High-resolution images with more overlap
- **Visualization**: Disabled
- **Directory Filter**: None
- **Slice Size**: 1920x1920 (larger)
- **Overlap**: 0.3 (30% - more thorough)
- **Use case**: Large, high-resolution images

## Configuration Parameters

### Required Parameters

```json
{
  "input_dir": "D:\\path\\to\\images",
  "output_file": "D:\\path\\to\\output\\detections.txt",
  "face_model": "D:\\path\\to\\face_model.pt",
  "plate_model": "D:\\path\\to\\plate_model.pt"
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_dir` | string | Root directory containing images (searched recursively) |
| `output_file` | string | Path where detection results will be saved |
| `face_model` | string | Path to face detection YOLO model (.pt file) |
| `plate_model` | string | Path to plate detection YOLO model (.pt file) |

### Optional Parameters

```json
{
  "device": "cuda:0",
  "slice_height": 1280,
  "slice_width": 1280,
  "overlap_ratio": 0.2,
  "visualize": 0,
  "visualization_dir": null,
  "resume": true,
  "dir_filter": null,
  "description": "My custom config"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | string | `"cuda:0"` | Device for inference: `"cuda:0"`, `"cuda:1"`, `"cpu"` |
| `slice_height` | integer | `1280` | Height of image slices in pixels |
| `slice_width` | integer | `1280` | Width of image slices in pixels |
| `overlap_ratio` | float | `0.2` | Overlap between slices (0.2 = 20%) |
| `visualize` | integer | `0` | Number of random images to visualize (0 = none) |
| `visualization_dir` | string/null | `null` | Directory to save visualizations (required if visualize > 0) |
| `resume` | boolean | `true` | Automatically resume from previous progress |
| `dir_filter` | string/null | `null` | Only process directories containing this text |
| `description` | string | `""` | Optional description of the config (for your reference) |

## Path Formatting

### Windows Paths
Use double backslashes `\\` or forward slashes `/`:

```json
// Correct - double backslashes
"input_dir": "D:\\images\\folder"

// Also correct - forward slashes
"input_dir": "D:/images/folder"

// WRONG - single backslash
"input_dir": "D:\images\folder"  // ❌ Will cause errors
```

### Absolute vs Relative Paths
Always use absolute paths for reliability:

```json
// ✓ Good - absolute path
"input_dir": "D:\\PLACE - Zotac\\BGD\\Testing\\images"

// ⚠️ Risky - relative path (depends on where you run the script)
"input_dir": "..\\Testing\\images"
```

## Common Configuration Scenarios

### Scenario 1: Quick Test Run

```json
{
  "description": "Test run with 10 visualizations",
  "input_dir": "D:\\test_images",
  "output_file": "D:\\output\\test.txt",
  "face_model": "D:\\models\\face.pt",
  "plate_model": "D:\\models\\plate.pt",
  "device": "cuda:0",
  "visualize": 10,
  "visualization_dir": "D:\\output\\test_vis",
  "slice_height": 1280,
  "slice_width": 1280,
  "overlap_ratio": 0.2
}
```

**Run:** `python sahi_dual_model.py --config config_test.json`

### Scenario 2: Process Only "Stitched" Folders

```json
{
  "description": "Only stitched directories",
  "input_dir": "D:\\all_images",
  "output_file": "D:\\output\\stitched_detections.txt",
  "face_model": "D:\\models\\face.pt",
  "plate_model": "D:\\models\\plate.pt",
  "dir_filter": "stitched",
  "visualize": 0
}
```

**Run:** `python sahi_dual_model.py --config config_stitched.json`

### Scenario 3: High-Resolution Images

```json
{
  "description": "High-res processing with more overlap",
  "input_dir": "D:\\highres_images",
  "output_file": "D:\\output\\highres_detections.txt",
  "face_model": "D:\\models\\face.pt",
  "plate_model": "D:\\models\\plate.pt",
  "slice_height": 1920,
  "slice_width": 1920,
  "overlap_ratio": 0.3,
  "visualize": 0
}
```

**Run:** `python sahi_dual_model.py --config config_highres.json`

### Scenario 4: CPU Processing (No GPU)

```json
{
  "description": "CPU processing for systems without GPU",
  "input_dir": "D:\\images",
  "output_file": "D:\\output\\detections.txt",
  "face_model": "D:\\models\\face.pt",
  "plate_model": "D:\\models\\plate.pt",
  "device": "cpu",
  "slice_height": 640,
  "slice_width": 640,
  "overlap_ratio": 0.1
}
```

**Note:** Smaller slices and less overlap for faster CPU processing.

### Scenario 5: Multiple GPU Setup

```json
{
  "description": "Use second GPU",
  "input_dir": "D:\\images",
  "output_file": "D:\\output\\detections.txt",
  "face_model": "D:\\models\\face.pt",
  "plate_model": "D:\\models\\plate.pt",
  "device": "cuda:1"
}
```

### Scenario 6: Production Run with Sampling

```json
{
  "description": "Full dataset with 100 visual samples",
  "input_dir": "D:\\production\\images",
  "output_file": "D:\\production\\output\\detections.txt",
  "face_model": "D:\\models\\face_v2.pt",
  "plate_model": "D:\\models\\plate_v3.pt",
  "device": "cuda:0",
  "visualize": 100,
  "visualization_dir": "D:\\production\\output\\samples",
  "dir_filter": "final",
  "slice_height": 1280,
  "slice_width": 1280,
  "overlap_ratio": 0.25
}
```

## Overriding Config Values

Command-line arguments override config file values:

```bash
# Use config but change device
python sahi_dual_model.py --config config.json --device cuda:1

# Use config but add visualization
python sahi_dual_model.py --config config.json --visualize 25 --visualization-dir D:\output\vis

# Use config but disable resume
python sahi_dual_model.py --config config.json --no-resume

# Multiple overrides
python sahi_dual_model.py --config config.json --device cuda:1 --visualize 50 --overlap-ratio 0.3
```

## Parameter Tuning Guide

### Slice Size (slice_height, slice_width)

| Size | Speed | Detection Quality | Use Case |
|------|-------|-------------------|----------|
| 640x640 | Fastest | Good for large objects | Low-res images, speed priority |
| 1280x1280 | Balanced | Good all-around | Default, works well for most cases |
| 1920x1920 | Slower | Best for small objects | High-res images, detail priority |
| 2560x2560 | Slowest | Excellent for tiny objects | Very high-res, maximum quality |

### Overlap Ratio

| Ratio | Speed | Detection Quality | Use Case |
|-------|-------|-------------------|----------|
| 0.1 (10%) | Fastest | May miss edge objects | Speed priority |
| 0.2 (20%) | Fast | Good balance | Default, recommended |
| 0.3 (30%) | Moderate | Better edge detection | High-quality results |
| 0.4 (40%) | Slow | Best edge detection | Maximum quality, slow |

### Visualization Count

| Count | Purpose |
|-------|---------|
| 0 | No visualization (fastest, for production) |
| 10 | Quick quality check |
| 50 | Medium sampling for evaluation |
| 100+ | Thorough quality assessment |

## Resume Feature

The script automatically saves progress. If interrupted:

1. **Just re-run the same command:**
   ```bash
   python sahi_dual_model.py --config config.json
   ```

2. **To start fresh, either:**
   - Set `"resume": false` in config, OR
   - Use `--no-resume` flag:
     ```bash
     python sahi_dual_model.py --config config.json --no-resume
     ```

3. **Progress file location:**
   - Saved at: `<output_file>_progress.txt`
   - Example: `detections_master.txt` → `detections_master_progress.txt`

## Troubleshooting Config Files

### Error: "Failed to load config file"
- Check that JSON syntax is valid (use a JSON validator)
- Ensure file path is correct
- Check for missing commas or quotes

### Error: "input_dir is required"
- Add required parameters to config file
- Or provide them via command line

### Error: "Invalid escape sequence"
```json
// Wrong
"input_dir": "D:\images\folder"  // ❌

// Correct
"input_dir": "D:\\images\\folder"  // ✓
"input_dir": "D:/images/folder"    // ✓
```

### Visualization not working
- Ensure `visualization_dir` is set when `visualize > 0`
- Check that the directory path is valid and writable

## Best Practices

1. **Create multiple configs** for different scenarios (test, production, etc.)
2. **Use descriptive names** for config files (config_test.json, config_production.json)
3. **Add descriptions** to configs for documentation
4. **Test first** with small dataset and visualization before full runs
5. **Keep a backup** of working configs
6. **Use absolute paths** to avoid path resolution issues
7. **Start with defaults** and adjust based on results

## Example: Creating Your Own Config

1. Copy the default config:
   ```bash
   copy config.json my_config.json
   ```

2. Edit with your paths:
   ```json
   {
     "description": "My custom detection run",
     "input_dir": "D:\\my_images",
     "output_file": "D:\\my_output\\detections.txt",
     "face_model": "D:\\my_models\\face_best.pt",
     "plate_model": "D:\\my_models\\plate_best.pt",
     "device": "cuda:0",
     "slice_height": 1280,
     "slice_width": 1280,
     "overlap_ratio": 0.2,
     "visualize": 20,
     "visualization_dir": "D:\\my_output\\vis",
     "resume": true,
     "dir_filter": null
   }
   ```

3. Test it:
   ```bash
   python sahi_dual_model.py --config my_config.json
   ```

4. Adjust parameters based on results and re-run.
