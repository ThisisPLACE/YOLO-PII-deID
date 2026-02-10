# YOLO Detection & Utilities

Lightweight repository of scripts and utilities for YOLO-based object detection, SAHI slicing, dataset creation, visualization, and Label Studio integration.

## What's changed

- `requirements.txt` now includes `torch` and `torchvision`. For CUDA-enabled installs, please use the official PyTorch selector to install the correct wheel for your GPU.
- A configuration template is available at `scripts/Sahi_detect/config_template.yaml` (copy and edit to create your runtime `config.yaml`).

## Installation

1. Create and activate a Python virtual environment:

```bash
python3 -m venv yolo_env
# Linux / macOS
source yolo_env/bin/activate
# Windows (PowerShell)
.\yolo_env\Scripts\Activate.ps1
```

2. Install PyTorch (choose proper wheel for your CUDA version). Example for CUDA via PyTorch's index (adjust to your CUDA):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

3. Install remaining dependencies:

```bash
pip install -r requirements.txt
```

Note: `requirements.txt` includes `torch`/`torchvision` as convenience, but installing PyTorch through the official selector ensures compatibility with your CUDA runtime.

## Quick Start

- Run a simple detection (example):

```bash
python yolo_detect_v11.py --input path/to/image.jpg --conf 0.5
```

- Run SAHI detection (example):

```bash
python scripts/Sahi_detect/sahi_detection.py --config scripts/Sahi_detect/config_template.yaml
```

## Configuration

Use `scripts/Sahi_detect/config_template.yaml` as a starting point. Copy it to `scripts/Sahi_detect/config.yaml` and update the `input_dir`, `output_dir`, and `model.path` values to point to your dataset and model weights.

## Files of Interest

- `yolo_detect_v11.py` — YOLO v11 inference script
- `scripts/train/yolo11_train.py` — Training script
- `scripts/Sahi_detect/` — SAHI detection scripts and configs
- `utils/` — utilities for staging images, visualization, dataset creation, Label Studio helpers
- `requirements.txt` — project dependencies

## Troubleshooting

- Verify GPU drivers and CUDA with `nvidia-smi` and test PyTorch with:

```bash
python -c "import torch; print('GPU Found!' if torch.cuda.is_available() else 'CPU Only')"
```

- If Pillow raises errors, see `scripts/Sahi_detect/INSTALL_PILLOW.md`.

## Contributing

Contributions welcome — open an issue or submit a PR.

---

**Last Updated:** February 2026
