"""
Analyze and visualize statistics from the dual-model detection output.
"""

import argparse
import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def load_detections(file_path: str) -> pd.DataFrame:
    """Load detection file into pandas DataFrame handling spaces in paths."""
    try:
        # Use r'\s+' as a separator to catch one or more spaces/tabs
        # Use engine='python' to handle complex separators more gracefully
        df = pd.read_csv(
            file_path,
            sep=r'\s+',
            engine='python',
            comment='#',
            names=['image_path', 'class_id', 'x_center', 'y_center', 'width', 'height', 'confidence'],
            on_bad_lines='warn' # This will skip lines that still fail and show a warning
        )
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise

def analyze_detections(df: pd.DataFrame, output_dir: str = None):
    """
    Generate comprehensive statistics and visualizations.
    
    Args:
        df: DataFrame with detections
        output_dir: Directory to save plots (optional)
    """
    
    print("=" * 80)
    print("DETECTION ANALYSIS REPORT")
    print("=" * 80)
    
    # Basic statistics
    total_detections = len(df)
    total_images = df['image_path'].nunique()
    face_detections = len(df[df['class_id'] == 0])
    plate_detections = len(df[df['class_id'] == 1])
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"{'─' * 80}")
    print(f"Total Images Processed: {total_images:,}")
    print(f"Total Detections: {total_detections:,}")
    print(f"  • Faces (class 0): {face_detections:,} ({face_detections/total_detections*100:.1f}%)")
    print(f"  • Plates (class 1): {plate_detections:,} ({plate_detections/total_detections*100:.1f}%)")
    
    # Detections per image
    detections_per_image = df.groupby('image_path').size()
    faces_per_image = df[df['class_id'] == 0].groupby('image_path').size()
    plates_per_image = df[df['class_id'] == 1].groupby('image_path').size()
    
    print(f"\n📈 DETECTIONS PER IMAGE")
    print(f"{'─' * 80}")
    print(f"Average detections per image: {detections_per_image.mean():.2f}")
    print(f"  • Faces: {faces_per_image.mean():.2f} per image")
    print(f"  • Plates: {plates_per_image.mean():.2f} per image")
    print(f"Max detections in single image: {detections_per_image.max()}")
    print(f"Images with no detections: {total_images - len(detections_per_image)}")
    
    # Confidence statistics
    print(f"\n🎯 CONFIDENCE SCORES")
    print(f"{'─' * 80}")
    print(f"Overall:")
    print(f"  • Mean: {df['confidence'].mean():.4f}")
    print(f"  • Median: {df['confidence'].median():.4f}")
    print(f"  • Min: {df['confidence'].min():.4f}")
    print(f"  • Max: {df['confidence'].max():.4f}")
    
    face_conf = df[df['class_id'] == 0]['confidence']
    plate_conf = df[df['class_id'] == 1]['confidence']
    
    print(f"\nFaces (class 0):")
    print(f"  • Mean: {face_conf.mean():.4f}")
    print(f"  • Median: {face_conf.median():.4f}")
    
    print(f"\nPlates (class 1):")
    print(f"  • Mean: {plate_conf.mean():.4f}")
    print(f"  • Median: {plate_conf.median():.4f}")
    
    # Confidence thresholds
    print(f"\n📊 DETECTIONS BY CONFIDENCE THRESHOLD")
    print(f"{'─' * 80}")
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        count = len(df[df['confidence'] >= thresh])
        pct = count / total_detections * 100
        print(f"Confidence ≥ {thresh:.1f}: {count:,} ({pct:.1f}%)")
    
    # Bounding box statistics
    print(f"\n📦 BOUNDING BOX STATISTICS")
    print(f"{'─' * 80}")
    print(f"Average box size (normalized):")
    print(f"  • Width: {df['width'].mean():.4f} (±{df['width'].std():.4f})")
    print(f"  • Height: {df['height'].mean():.4f} (±{df['height'].std():.4f})")
    
    # Top images with most detections
    print(f"\n🏆 TOP 10 IMAGES WITH MOST DETECTIONS")
    print(f"{'─' * 80}")
    top_images = detections_per_image.nlargest(10)
    for img_path, count in top_images.items():
        img_name = Path(img_path).name
        faces = len(df[(df['image_path'] == img_path) & (df['class_id'] == 0)])
        plates = len(df[(df['image_path'] == img_path) & (df['class_id'] == 1)])
        print(f"{img_name}: {count} detections ({faces} faces, {plates} plates)")
    
    # Directory analysis
    print(f"\n📁 DIRECTORY ANALYSIS")
    print(f"{'─' * 80}")
    df['directory'] = df['image_path'].apply(lambda x: str(Path(x).parent))
    dir_stats = df.groupby('directory').size().sort_values(ascending=False)
    print(f"Total directories: {len(dir_stats)}")
    print(f"\nTop 10 directories by detection count:")
    for dir_path, count in dir_stats.head(10).items():
        dir_name = Path(dir_path).name
        print(f"  {dir_name}: {count:,} detections")
    
    # Generate visualizations if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 GENERATING VISUALIZATIONS...")
        print(f"{'─' * 80}")
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Confidence distribution by class
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(face_conf, bins=50, alpha=0.7, label='Faces', color='blue', edgecolor='black')
        axes[0].axvline(face_conf.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {face_conf.mean():.3f}')
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Face Detection Confidence Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(plate_conf, bins=50, alpha=0.7, label='Plates', color='green', edgecolor='black')
        axes[1].axvline(plate_conf.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {plate_conf.mean():.3f}')
        axes[1].set_xlabel('Confidence Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Plate Detection Confidence Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        conf_dist_path = output_path / 'confidence_distribution.png'
        plt.savefig(conf_dist_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {conf_dist_path}")
        plt.close()
        
        # 2. Detections per image histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(detections_per_image, bins=min(50, detections_per_image.max()), 
                color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(detections_per_image.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {detections_per_image.mean():.2f}')
        ax.set_xlabel('Detections per Image')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Detections per Image')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        det_per_img_path = output_path / 'detections_per_image.png'
        plt.savefig(det_per_img_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {det_per_img_path}")
        plt.close()
        
        # 3. Class distribution pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ['#3498db', '#2ecc71']
        labels = ['Faces', 'Plates']
        sizes = [face_detections, plate_detections]
        explode = (0.05, 0.05)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontsize': 14, 'weight': 'bold'})
        ax.set_title('Detection Class Distribution', fontsize=16, weight='bold')
        plt.tight_layout()
        class_dist_path = output_path / 'class_distribution.png'
        plt.savefig(class_dist_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {class_dist_path}")
        plt.close()
        
        # 4. Bounding box size scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        face_data = df[df['class_id'] == 0]
        plate_data = df[df['class_id'] == 1]
        
        ax.scatter(face_data['width'], face_data['height'], alpha=0.3, 
                   label='Faces', color='blue', s=10)
        ax.scatter(plate_data['width'], plate_data['height'], alpha=0.3, 
                   label='Plates', color='green', s=10)
        ax.set_xlabel('Box Width (normalized)')
        ax.set_ylabel('Box Height (normalized)')
        ax.set_title('Bounding Box Size Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        bbox_path = output_path / 'bbox_size_distribution.png'
        plt.savefig(bbox_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {bbox_path}")
        plt.close()
        
        print(f"\n✓ All visualizations saved to: {output_dir}")
    
    print("\n" + "=" * 80)


def export_filtered_detections(df: pd.DataFrame, output_file: str, 
                                min_confidence: float = None,
                                class_filter: int = None):
    """
    Export filtered detections to a new file.
    
    Args:
        df: DataFrame with detections
        output_file: Output file path
        min_confidence: Minimum confidence threshold
        class_filter: Filter by class ID (0=faces, 1=plates)
    """
    filtered_df = df.copy()
    
    if min_confidence is not None:
        filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
    
    if class_filter is not None:
        filtered_df = filtered_df[filtered_df['class_id'] == class_filter]
    
    # Export
    with open(output_file, 'w') as f:
        f.write("# image_path class_id x_center y_center width height confidence\n")
        for _, row in filtered_df.iterrows():
            f.write(
                f"{row['image_path']} {row['class_id']} "
                f"{row['x_center']:.6f} {row['y_center']:.6f} "
                f"{row['width']:.6f} {row['height']:.6f} "
                f"{row['confidence']:.6f}\n"
            )
    
    print(f"\n✓ Exported {len(filtered_df)} detections to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze dual-model detection results'
    )
    
    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Input detection file to analyze'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_output',
        help='Directory to save visualizations (default: analysis_output)'
    )
    parser.add_argument(
        '--export-filtered',
        type=str,
        default=None,
        help='Export filtered detections to this file'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=None,
        help='Filter by minimum confidence (used with --export-filtered)'
    )
    parser.add_argument(
        '--class-filter',
        type=int,
        choices=[0, 1],
        default=None,
        help='Filter by class: 0=faces, 1=plates (used with --export-filtered)'
    )
    
    args = parser.parse_args()
    
    # Load detections
    print(f"Loading detections from: {args.input_file}")
    df = load_detections(args.input_file)
    print(f"✓ Loaded {len(df)} detections from {df['image_path'].nunique()} images\n")
    
    # Analyze
    output_dir = args.output_dir if args.visualize else None
    analyze_detections(df, output_dir)
    
    # Export filtered if requested
    if args.export_filtered:
        export_filtered_detections(
            df,
            args.export_filtered,
            args.min_confidence,
            args.class_filter
        )


if __name__ == "__main__":
    main()
