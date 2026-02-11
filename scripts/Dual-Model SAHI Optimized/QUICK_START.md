# Quick Start Guide - Which Version to Use?

## 🚀 Fast Decision Tree

```
How many images do you have?
│
├─ < 500 images
│  └─ Use: sahi_dual_model.py (Original)
│     Config: config.json
│     Speed: Good enough for small datasets
│
├─ 500 - 5,000 images
│  └─ Use: sahi_dual_model_optimized.py (Optimized)
│     Config: config_optimized.json
│     Speed: 2-3x faster
│
└─ > 5,000 images
   └─ Use: sahi_dual_model_optimized.py (Optimized)
      │
      ├─ Single GPU
      │  Config: config_optimized.json
      │  Speed: 2-3x faster
      │
      └─ Multiple GPUs
         Config: config_multigpu.json
         Speed: 4-6x faster
```

## ⚡ Quick Commands

### Original Version (Simple, Reliable)
```bash
# Edit config.json with your paths first
python sahi_dual_model.py --config config.json
```

### Optimized Version - Single GPU
```bash
# Edit config_optimized.json with your paths first
python sahi_dual_model_optimized.py --config config_optimized.json
```

### Optimized Version - Multi GPU
```bash
# Edit config_multigpu.json with your paths first
python sahi_dual_model_optimized.py --config config_multigpu.json
```

## 📋 Configuration Checklist

### Before Running, Edit Your Config File:

1. ✅ `input_dir` - Where your images are
2. ✅ `output_file` - Where to save results
3. ✅ `face_model` - Path to face detection model
4. ✅ `plate_model` - Path to plate detection model
5. ⚙️ `devices` - GPU(s) to use (optimized only)
6. ⚙️ `num_workers` - Number of threads (optimized only)

## 🎯 Recommended Settings

### For Testing (10-100 images)
```json
{
  "visualize": 10,
  "num_workers": 2
}
```
**Command**: `python sahi_dual_model.py --config config_test.json`

### For Production - Single GPU
```json
{
  "devices": ["cuda:0"],
  "num_workers": 4,
  "visualize": 0
}
```
**Command**: `python sahi_dual_model_optimized.py --config config_optimized.json`

### For Production - Dual GPU
```json
{
  "devices": ["cuda:0", "cuda:1"],
  "num_workers": 8,
  "visualize": 0
}
```
**Command**: `python sahi_dual_model_optimized.py --config config_multigpu.json`

## 🔧 Tuning num_workers

### Rule of Thumb
```
num_workers = num_gpus × 2 to 4
```

### Examples:
- 1 GPU → `num_workers: 4`
- 2 GPUs → `num_workers: 8`
- 4 GPUs → `num_workers: 16`

### If You See:
- **GPU utilization < 50%** → Increase num_workers
- **Out of memory errors** → Decrease num_workers
- **CPU at 100%** → Decrease num_workers

## 📊 Expected Performance

| Dataset Size | Version | Config | Expected Time* |
|--------------|---------|--------|----------------|
| 100 images | Original | config.json | ~5 min |
| 100 images | Optimized | config_optimized.json | ~2 min |
| 1,000 images | Original | config.json | ~50 min |
| 1,000 images | Optimized (1 GPU) | config_optimized.json | ~20 min |
| 1,000 images | Optimized (2 GPU) | config_multigpu.json | ~10 min |
| 10,000 images | Original | config.json | ~8 hours |
| 10,000 images | Optimized (1 GPU) | config_optimized.json | ~3 hours |
| 10,000 images | Optimized (2 GPU) | config_multigpu.json | ~1.5 hours |

*Based on RTX 3090, 4K images, default settings

## 🛠️ Common Issues & Fixes

### Issue: Script very slow
**Fix**: Use optimized version with more workers
```bash
python sahi_dual_model_optimized.py --config config_optimized.json --num-workers 6
```

### Issue: Out of memory
**Fix**: Reduce workers or slice size
```bash
python sahi_dual_model_optimized.py --config config_optimized.json --num-workers 2 --slice-height 640
```

### Issue: Want to use only "stitched" folders
**Fix**: Add dir_filter to config or use flag
```bash
python sahi_dual_model.py --config config.json --dir-filter "stitched"
```

### Issue: Processing interrupted
**Fix**: Just re-run same command (auto-resumes)
```bash
# Will continue from where it stopped
python sahi_dual_model_optimized.py --config config_optimized.json
```

### Issue: Want to start fresh
**Fix**: Use --no-resume flag
```bash
python sahi_dual_model.py --config config.json --no-resume
```

## 📁 File Organization

```
your_project/
├── sahi_dual_model.py              # Original version
├── sahi_dual_model_optimized.py   # Optimized version
├── config.json                     # Original config
├── config_optimized.json           # Optimized single GPU
├── config_multigpu.json            # Optimized multi GPU
├── config_test.json                # Test with visualization
├── README.md                       # Full documentation
├── OPTIMIZATION_GUIDE.md           # Performance tuning
├── VERSION_COMPARISON.md           # Choose version
└── analyze_detections.py           # Analyze results
```

## 🎓 Learning Path

### Step 1: Start Simple (5 minutes)
```bash
# Edit config_test.json with your paths
python sahi_dual_model.py --config config_test.json
# Check the 10 visualizations
```

### Step 2: Test on Small Sample (30 minutes)
```bash
# Edit config.json with your paths
python sahi_dual_model.py --config config.json --visualize 50
# Review results, adjust parameters if needed
```

### Step 3: Full Production Run
```bash
# If < 1000 images
python sahi_dual_model.py --config config.json

# If > 1000 images, use optimized
python sahi_dual_model_optimized.py --config config_optimized.json
```

### Step 4: Analyze Results
```bash
python analyze_detections.py --input-file output/detections.txt --visualize
```

## 💡 Pro Tips

1. **Always test first** with 10-100 images and visualization
2. **Monitor GPU** with `nvidia-smi -l 1` during processing
3. **Use resume feature** for large datasets (interruption-safe)
4. **Batch by directory** using `--dir-filter` for organized processing
5. **Keep configs** for different scenarios (test, production, highres)

## 🆘 Need Help?

### Check GPU availability
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### Test config file
```bash
python -c "import json; print(json.load(open('config.json')))"
```

### See all options
```bash
python sahi_dual_model.py --help
python sahi_dual_model_optimized.py --help
```

## 📚 Additional Resources

- **README.md** - Complete documentation
- **CONFIG_GUIDE.md** - Configuration deep dive
- **OPTIMIZATION_GUIDE.md** - Performance tuning
- **VERSION_COMPARISON.md** - Detailed comparison

---

## Summary: Your First Run

1. **Edit** `config.json` with your paths
2. **Test** with 10 images:
   ```bash
   python sahi_dual_model.py --config config.json --visualize 10
   ```
3. **Review** visualizations in output directory
4. **Run full** dataset:
   ```bash
   # Small dataset
   python sahi_dual_model.py --config config.json
   
   # Large dataset
   python sahi_dual_model_optimized.py --config config_optimized.json
   ```
5. **Analyze** results:
   ```bash
   python analyze_detections.py --input-file output/detections.txt
   ```

That's it! 🎉
