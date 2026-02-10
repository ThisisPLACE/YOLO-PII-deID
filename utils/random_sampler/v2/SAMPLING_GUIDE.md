# Random Sampling Feature Guide

## Overview

The enhanced script now includes **random sampling** functionality to process a subset of your images. This is useful for testing, validation, or when you only want to process a representative sample.

---

## Quick Usage

### Option 1: Sample Fixed Number of Images
```bash
# Process exactly 100 random images
python script.py /input /output --sample-size 100

# Process exactly 50 random images
python script.py /input /output --sample-size 50
```

### Option 2: Sample Percentage of Images
```bash
# Process 10% of all images
python script.py /input /output --sample-percent 10

# Process 5% of images
python script.py /input /output --sample-percent 5

# Process 25% of images
python script.py /input /output --sample-percent 25
```

### Option 3: Reproducible Sampling (Same Images Every Time)
```bash
# Always select the same 100 images
python script.py /input /output --sample-size 100 --random-seed 42

# Later, run with same seed to get identical sample
python script.py /input /output --sample-size 100 --random-seed 42
# Will process the exact same 100 images
```

---

## Common Use Cases

### Testing Quality Settings
```bash
# Test a new quality preset on 50 random images first
python script.py /input /output --sample-size 50 --preset balanced --dry-run

# Review the output, then process entire batch
python script.py /input /output --preset balanced
```

### Validation & QA
```bash
# Quality assurance: check 5% of processed images
python script.py /input /output --sample-percent 5 --preset balanced --manifest qa_report.json

# Check results in manifest before processing entire batch
```

### Memory-Constrained Systems
```bash
# System has limited memory? Process only 10%
python script.py /input /output --sample-percent 10 --preset balanced

# Then run again with different seed to process more
python script.py /input /output --sample-percent 10 --random-seed 123
```

### A/B Testing
```bash
# Test version A: 100 images with preset "fast"
python script.py /input /output_fast --sample-size 100 --random-seed 1 --preset fast

# Test version B: same 100 images with preset "balanced"
python script.py /input /output_balanced --sample-size 100 --random-seed 1 --preset balanced

# Compare results
```

### Dataset Sampling
```bash
# Create representative sample for review
python script.py /input /output --sample-percent 1 --manifest sample_stats.json

# Process only 1% of large dataset (e.g., 100 images from 10,000)
```

---

## How Random Sampling Works

### Selection Process
1. Script scans all images in directory
2. Finds all that match filters (if any)
3. Randomly selects requested number/percentage
4. Processes only selected images

### Randomness
- Selection is truly random (using Python's random module)
- Each run produces different sample (unless --random-seed used)

### Reproducibility
```bash
# Same seed = Same images selected (for testing/debugging)
python script.py /input /output --sample-size 100 --random-seed 42
# Later:
python script.py /input /output --sample-size 100 --random-seed 42
# Same 100 images processed both times
```

---

## Sampling with Filters

Sampling works AFTER filtering. So:

```bash
# Get 5% of images that are larger than 1MB
python script.py /input /output \
  --min-size 1MB \
  --sample-percent 5

# Results in manifest will show actual sampling done
```

Order of operations:
1. Find all images
2. Apply filters (size, dimensions, date, etc.)
3. Randomly sample from filtered results
4. Process sampled images

---

## Output & Reporting

### What Gets Reported
```bash
python script.py /input /output --sample-size 100 --manifest report.json
```

The manifest includes sampling information:
```json
{
  "sampling": {
    "total_available": 5000,
    "sampled": 100,
    "sample_percent": 2.0
  },
  "summary": {
    "total_files": 100,
    ...
  }
}
```

---

## Examples

### Example 1: Quick Test
```bash
# Test new settings on small sample
python script.py /photos /output --sample-size 20 --preset balanced --dry-run

# Review it looks good

# Process all images with same settings
python script.py /photos /output --preset balanced
```

### Example 2: Validation Workflow
```bash
# Process 10% for validation
python script.py /photos /output_validation \
  --sample-percent 10 \
  --preset balanced \
  --manifest validation_report.json

# Check validation_report.json

# If good, process remaining 90%
python script.py /photos /output_final \
  --preset balanced \
  --manifest final_report.json
```

### Example 3: Consistent Sampling
```bash
# First engineer tests
python script.py /photos /review \
  --sample-size 100 \
  --random-seed 12345 \
  --manifest eng_review.json

# Client tests same images for consistency
python script.py /photos /review \
  --sample-size 100 \
  --random-seed 12345 \
  --manifest client_review.json

# Both reviewed same 100 images (deterministic sampling)
```

### Example 4: Large Batch Processing
```bash
# 100,000 images, test on 1%
python script.py /massive_folder /output \
  --sample-percent 1 \
  --preset fast \
  --manifest sample_1percent.json

# Review results

# Process more samples
python script.py /massive_folder /output \
  --sample-percent 1 \
  --random-seed 2 \
  --preset fast

# Finally full batch
python script.py /massive_folder /output \
  --preset fast \
  --manifest final_batch.json
```

---

## Arguments Reference

### Sampling Arguments

| Argument | Use | Example |
|----------|-----|---------|
| `--sample-size N` | Process exactly N images | `--sample-size 100` |
| `--sample-percent X` | Process X% of images | `--sample-percent 10` |
| `--random-seed N` | Make sampling reproducible | `--random-seed 42` |

### Important Notes

- **Cannot use both** `--sample-size` AND `--sample-percent` together
- **Sampling is AFTER filtering**, so combine with other options
- **Random seed** only matters if you want reproducible results
- **Minimum 1 image** always selected (even with small percentages)

---

## Sampling Size Decision Matrix

| Scenario | Recommendation | Example |
|----------|---|---------|
| Quick test | 10-50 images | `--sample-size 30` |
| Quality validation | 1-5% | `--sample-percent 3` |
| Confidence check | 5% | `--sample-percent 5` |
| Small dataset (< 100) | 50-80% | `--sample-percent 70` |
| Large dataset (1000+) | 1-3% | `--sample-percent 2` |
| Processing with limited RAM | 5-10% | `--sample-percent 8` |

---

## Real-World Workflows

### Workflow 1: New Feature Testing
```bash
# 1. Test on small sample
python script.py /input /test --sample-size 50 --preset webp-balanced --dry-run

# 2. Check output quality
ls -lh /test/ | head

# 3. If good, process all
python script.py /input /output --preset webp-balanced

# 4. Generate report
python script.py /input /output --preset webp-balanced --manifest final_report.json
```

### Workflow 2: QA/Validation
```bash
# 1. Generate sample for QA
python script.py /input /qa_sample \
  --sample-percent 5 \
  --preset balanced \
  --manifest qa_manifest.json

# 2. QA team reviews qa_sample folder and qa_manifest.json

# 3. If approved, process full batch
python script.py /input /production \
  --preset balanced \
  --manifest production_manifest.json

# 4. Compare statistics
# QA manifest shows 5% sample quality
# Production manifest shows full batch quality
```

### Workflow 3: Benchmarking
```bash
# Test different presets on same sample
python script.py /input /test_fast \
  --sample-size 100 \
  --random-seed 999 \
  --preset fast \
  --manifest fast.json

python script.py /input /test_balanced \
  --sample-size 100 \
  --random-seed 999 \
  --preset balanced \
  --manifest balanced.json

python script.py /input /test_high \
  --sample-size 100 \
  --random-seed 999 \
  --preset high \
  --manifest high.json

# Compare results in: fast.json, balanced.json, high.json
# Same 100 images (reproducible seed), different settings
```

---

## Tips & Tricks

### Tip 1: Combine with Dry-Run for Safe Testing
```bash
python script.py /input /output \
  --sample-percent 5 \
  --dry-run \
  --manifest preview.json
# Preview 5% without actually processing
```

### Tip 2: Use with Filters for Targeted Sampling
```bash
# Sample from only large images
python script.py /input /output \
  --min-size 10MB \
  --sample-percent 20

# Sample from only recent images
python script.py /input /output \
  --modified-after 2024-01-01 \
  --sample-percent 10
```

### Tip 3: Multiple Runs for Better Coverage
```bash
# Run multiple times with different seeds
for seed in 1 2 3 4 5; do
  python script.py /input /output_$seed \
    --sample-percent 10 \
    --random-seed $seed \
    --manifest report_$seed.json
done

# Processed 5 different 10% samples (50% total coverage)
```

### Tip 4: Keep Seed for Documentation
```bash
# Document which seed was used
python script.py /input /output \
  --sample-size 100 \
  --random-seed 42 \
  --log-file "sample_42.log"

# Later: "This output was created with seed 42"
# Can reproduce with: --random-seed 42
```

---

## Combining Sampling with Other Features

### Sampling + Quality Settings
```bash
python script.py /input /output \
  --sample-size 100 \
  --preset webp-balanced \
  --quality 75 \
  --resize 80
```

### Sampling + Filtering
```bash
python script.py /input /output \
  --sample-percent 5 \
  --min-width 1920 \
  --min-size 5MB \
  --modified-after 2024-01-01
```

### Sampling + EXIF Handling
```bash
python script.py /input /output \
  --sample-percent 10 \
  --preserve-exif \
  --extract-exif
```

### Sampling + Reporting
```bash
python script.py /input /output \
  --sample-size 50 \
  --manifest report.json \
  --log-file process.log
```

---

## Troubleshooting

### Q: All images selected, not a sample?
A: Check command syntax:
```bash
# Wrong: No sample specified
python script.py /input /output

# Right: Specify sample
python script.py /input /output --sample-size 100
```

### Q: Different images selected each run?
A: This is normal (random selection). To get same images:
```bash
# Add --random-seed
python script.py /input /output --sample-size 100 --random-seed 42
```

### Q: Sample size larger than available images?
A: Script automatically caps to total found:
```bash
# If only 50 images exist and you request 100
python script.py /input /output --sample-size 100
# Will process all 50 available
```

### Q: No images selected?
A: Check filters:
```bash
# Filters might exclude everything
python script.py /input /output --sample-percent 10 --min-size 100MB

# Verify with dry-run first
python script.py /input /output --sample-percent 10 --dry-run
```

---

## Summary

**Random Sampling Allows:**
- ✅ Test on subset before full processing
- ✅ Representative sampling for validation
- ✅ Reproducible samples with seeds
- ✅ Control processing on resource-limited systems
- ✅ A/B testing with same images
- ✅ Efficient QA workflows

**Key Syntax:**
- `--sample-size 100` → Process exactly 100 random images
- `--sample-percent 10` → Process 10% of images
- `--random-seed 42` → Make selection reproducible

