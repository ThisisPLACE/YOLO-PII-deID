#!/usr/bin/env python3
"""
Quick model inspector - Display classes from a .pt model
Usage: python inspect_model_simple.py <path_to_model.pt> [device]
Example: python inspect_model_simple.py runs/detect/v11_high_res_mix/weights/best.pt cuda:0
"""

import sys
from pathlib import Path
from sahi import AutoDetectionModel

def inspect(model_path, device='cuda:0'):
    """Load and display classes from a model."""
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    print(f"\n📦 Loading model: {model_path.name}")
    print(f"🖥️  Device: {device}\n")
    
    try:
        model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=str(model_path),
            device=device
        )
        print("✅ Model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Display classes
    mapping = model.category_mapping
    
    if not mapping:
        print("❌ No classes found in model")
        sys.exit(1)
    
    # Determine format and display
    first_key = next(iter(mapping.keys()))
    first_value = next(iter(mapping.values()))
    
    if isinstance(first_key, int) or (isinstance(first_key, str) and first_key.isdigit()):
        # {id: name} format
        items = sorted(mapping.items(), key=lambda x: int(x[0]) if isinstance(x[0], str) else x[0])
    else:
        # {name: id} format - sort by ID value
        items = sorted(mapping.items(), key=lambda x: int(x[1]) if isinstance(x[1], str) else x[1])
        items = [(str(cid), cname) for cname, cid in items]
    
    # Display in tabular format
    print("\n" + "=" * 60)
    print("AVAILABLE CLASSES".center(60))
    print("=" * 60)
    print(f"{'Class ID':<12} | {'Class Name':<44}")
    print("-" * 60)
    
    for cid, cname in items:
        cid_str = str(cid)
        print(f"{cid_str:<12} | {cname:<44}")
    
    print("=" * 60)
    print(f"Total classes: {len(mapping)}".rjust(60))
    print("=" * 60 + "\n")
    
    # Print for copy-paste into config
    print("📋 For sahi_config.py:")
    print("-" * 50)
    
    if isinstance(first_key, int) or (isinstance(first_key, str) and first_key.isdigit()):
        class_names = list(mapping.values())
    else:
        class_names = sorted(mapping.keys(), key=lambda x: int(mapping[x]) if isinstance(mapping[x], str) else mapping[x])
    
    print(f"CLASSES_TO_DETECT = {class_names}")
    print("-" * 50)
    print()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    model_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else 'cuda:0'
    
    inspect(model_path, device)