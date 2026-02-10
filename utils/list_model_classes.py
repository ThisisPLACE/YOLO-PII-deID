#!/usr/bin/env python3
"""
Display all available classes in yolov8m-oiv7.pt model
"""

from ultralytics import YOLO
import argparse

def list_model_classes(model_path):
    """Load model and display all available classes."""
    
    print(f"\nLoading model: {model_path}")
    print("-" * 80)
    
    try:
        model = YOLO(model_path)
        
        print(f"\nModel: {model.names}")
        print("\n" + "=" * 80)
        print("AVAILABLE CLASSES")
        print("=" * 80)
        print(f"\nTotal classes: {len(model.names)}\n")
        print(f"{'Class ID':<12} | {'Class Name':<30} | Notes")
        print("-" * 80)
        
        for class_id, class_name in model.names.items():
            # Mark faces and license plates
            notes = ""
            if 'face' in class_name.lower():
                notes = "← FACE DETECTION"
            elif 'license' in class_name.lower() or 'plate' in class_name.lower():
                notes = "← LICENSE PLATE"
            
            print(f"{class_id:<12} | {class_name:<30} | {notes}")
        
        print("\n" + "=" * 80)
        print("\nUSAGE EXAMPLES:")
        print("=" * 80)
        
        print("\n1. Using class names in config:")
        print('   "target_classes": ["face", "license plate"]')
        
        print("\n2. Using class IDs in config:")
        print('   "target_classes": ["1", "2"]')
        
        print("\n3. Mixed names and IDs in config:")
        print('   "target_classes": ["face", "2"]')
        
        print("\n4. Command line examples:")
        print('   python Sahi_FN_detection.py --target-classes face')
        print('   python Sahi_FN_detection.py --target-classes 1 2')
        print('   python Sahi_FN_detection.py --target-classes face 2')
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Make sure the model file exists at: {model_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Display available classes in YOLO model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python list_model_classes.py
  python list_model_classes.py --model-path custom_model.pt
        """
    )
    
    parser.add_argument('--model-path', type=str, default='.pt/yolov8m-oiv7.pt',
                        help='Path to YOLO model file (default: .pt/yolov8m-oiv7.pt)')
    
    args = parser.parse_args()
    
    list_model_classes(args.model_path)

if __name__ == "__main__":
    main()
