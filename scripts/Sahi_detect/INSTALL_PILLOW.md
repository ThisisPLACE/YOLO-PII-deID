# Installing Pillow for JPG Conversion

JPG conversion requires Pillow (PIL) library. Follow these simple steps.

## Quick Install

### Windows (Command Prompt or PowerShell):
```bash
pip install Pillow
```

### Linux/Mac:
```bash
pip install Pillow
```

### With Anaconda:
```bash
conda install Pillow
```

## Verify Installation

Check that Pillow is installed correctly:

```bash
python -c "from PIL import Image; print('Pillow OK')"
```

**Expected output:**
```
Pillow OK
```

## Troubleshooting

### "pip: command not found"

Make sure Python is installed and in your PATH.

**Windows:**
```bash
python -m pip install Pillow
```

**Linux/Mac:**
```bash
python3 -m pip install Pillow
```

### "Permission denied"

Add `--user` flag:
```bash
pip install --user Pillow
```

### Using Anaconda Environment

Make sure your environment is activated:

```bash
# Activate environment
conda activate your_env_name

# Install Pillow
conda install Pillow
```

## Quick Test

Create a test file `test_pillow.py`:

```python
from PIL import Image
import tempfile
from pathlib import Path

# Create a test PNG
img = Image.new('RGB', (640, 480), color='red')
with tempfile.TemporaryDirectory() as tmpdir:
    png_path = Path(tmpdir) / "test.png"
    jpg_path = Path(tmpdir) / "test.jpg"
    
    img.save(png_path)
    print(f"✓ Created test PNG: {png_path.stat().st_size} bytes")
    
    img.save(jpg_path, 'JPEG', quality=85)
    print(f"✓ Created test JPG: {jpg_path.stat().st_size} bytes")

print("\n✓ Pillow is working correctly!")
```

Run it:
```bash
python test_pillow.py
```

**Expected output:**
```
✓ Created test PNG: 1234 bytes
✓ Created test JPG: 567 bytes
✓ Pillow is working correctly!
```

## Next Steps

Once Pillow is installed:

1. Update `config.yaml`:
```yaml
output:
  visual_format: "jpg"
  jpg_quality: 85
```

2. Run the detection script:
```bash
python sahi_detection.py
```

JPG conversion will work automatically!
