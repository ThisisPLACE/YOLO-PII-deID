import os
from pathlib import Path
from sahi import AutoDetectionModel
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def inspect_model(model_path, model_type='yolov8', device='cuda:0'):
    """
    Load a .pt model and display all available classes.
    
    Args:
        model_path: Path to the .pt model file
        model_type: Type of model (e.g., 'yolov8')
        device: Device to use ('cuda:0', 'cpu', etc.)
    """
    
    # Validate model path
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        logger.error(f"Model file not found: {model_path}")
        return None
    
    if not model_path_obj.suffix == '.pt':
        logger.warning(f"File is not .pt format: {model_path_obj.suffix}")
    
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Device: {device}")
    
    try:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=model_path,
            device=device
        )
        logger.info("✓ Model loaded successfully\n")
        
        # Get category mapping
        category_mapping = detection_model.category_mapping
        
        if not category_mapping:
            logger.warning("No category mapping found in model")
            return None
        
        # Display classes
        logger.info("="*60)
        logger.info("AVAILABLE CLASSES")
        logger.info("="*60)
        
        # Check format of category_mapping
        if isinstance(category_mapping, dict):
            # Try to determine if it's {id: name} or {name: id}
            first_key = next(iter(category_mapping.keys()))
            
            if isinstance(first_key, int):
                # {id: name} format
                logger.info(f"Total classes: {len(category_mapping)}\n")
                logger.info("Format: Class ID -> Class Name")
                logger.info("-"*60)
                for class_id, class_name in sorted(category_mapping.items()):
                    logger.info(f"  ID {class_id:2d}: {class_name}")
            else:
                # {name: id} format
                logger.info(f"Total classes: {len(category_mapping)}\n")
                logger.info("Format: Class Name -> Class ID")
                logger.info("-"*60)
                for class_name, class_id in sorted(category_mapping.items(), key=lambda x: x[1]):
                    logger.info(f"  ID {class_id:2d}: {class_name}")
        
        logger.info("="*60)
        logger.info(f"\nTotal classes found: {len(category_mapping)}")
        
        # Return the category mapping for further use
        return category_mapping
        
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        logger.error(f"Make sure the model path is correct and model type is '{model_type}'")
        return None

def batch_inspect_models(models_dir, model_type='yolov8', device='cuda:0'):
    """
    Inspect all .pt models in a directory.
    
    Args:
        models_dir: Directory containing .pt model files
        model_type: Type of model
        device: Device to use
    """
    models_dir_obj = Path(models_dir)
    
    if not models_dir_obj.exists():
        logger.error(f"Directory not found: {models_dir}")
        return
    
    # Find all .pt files
    pt_files = list(models_dir_obj.rglob('*.pt'))
    
    if not pt_files:
        logger.warning(f"No .pt files found in {models_dir}")
        return
    
    logger.info(f"Found {len(pt_files)} .pt files\n")
    
    results = {}
    
    for idx, model_path in enumerate(pt_files, 1):
        logger.info(f"\n[{idx}/{len(pt_files)}] {model_path.name}")
        logger.info("-"*60)
        
        category_mapping = inspect_model(str(model_path), model_type, device)
        results[str(model_path)] = category_mapping
        
        logger.info("")
    
    return results

def export_classes_to_file(category_mapping, output_file='model_classes.txt'):
    """
    Export available classes to a text file.
    
    Args:
        category_mapping: Dictionary of class mappings
        output_file: Output file path
    """
    if not category_mapping:
        logger.error("No category mapping to export")
        return
    
    try:
        with open(output_file, 'w') as f:
            f.write("Available Classes\n")
            f.write("="*50 + "\n\n")
            
            if isinstance(category_mapping, dict):
                first_key = next(iter(category_mapping.keys()))
                
                if isinstance(first_key, int):
                    # {id: name} format
                    for class_id, class_name in sorted(category_mapping.items()):
                        f.write(f"ID {class_id}: {class_name}\n")
                else:
                    # {name: id} format
                    for class_name, class_id in sorted(category_mapping.items(), key=lambda x: x[1]):
                        f.write(f"ID {class_id}: {class_name}\n")
        
        logger.info(f"✓ Classes exported to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to export classes: {e}")

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("MODEL CLASS INSPECTOR")
    print("="*60 + "\n")
    
    # Check if model path is provided as command line argument
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        device = sys.argv[2] if len(sys.argv) > 2 else 'cuda:0'
        
        logger.info(f"Inspecting single model...")
        category_mapping = inspect_model(model_path, device=device)
        
        # Optionally export to file
        if category_mapping:
            export_choice = input("\nExport classes to file? (y/n): ").lower()
            if export_choice == 'y':
                output_file = input("Output filename (default: model_classes.txt): ") or "model_classes.txt"
                export_classes_to_file(category_mapping, output_file)
    
    else:
        # Interactive mode
        print("Options:")
        print("  1. Inspect a single model")
        print("  2. Inspect all models in a directory")
        print("  3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            model_path = input("\nEnter model path (.pt file): ").strip()
            device = input("Device (default: cuda:0): ").strip() or 'cuda:0'
            
            category_mapping = inspect_model(model_path, device=device)
            
            if category_mapping:
                export_choice = input("\nExport classes to file? (y/n): ").lower()
                if export_choice == 'y':
                    output_file = input("Output filename (default: model_classes.txt): ") or "model_classes.txt"
                    export_classes_to_file(category_mapping, output_file)
        
        elif choice == '2':
            models_dir = input("\nEnter directory path: ").strip()
            device = input("Device (default: cuda:0): ").strip() or 'cuda:0'
            
            results = batch_inspect_models(models_dir, device=device)
        
        elif choice == '3':
            logger.info("Exiting...")
        
        else:
            logger.error("Invalid option")
