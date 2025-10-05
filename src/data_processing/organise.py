import os  
import cv2  
import numpy as np 
import random  
from tqdm import tqdm  
from collections import Counter 
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

INPUT_DIR = r"C:\Sem 7\LY Project\Project\Critical-AI\chest_xray\chest_xray"
OUTPUT_DIR = r"C:\Sem 7\LY Project\Project\Critical-AI\chest_xray_split"
SPLITS = ["train", "val", "test"]

# Quality control parameters
MIN_RESOLUTION = (400, 400)
MIN_MEAN_INTENSITY = 10
MAX_MEAN_INTENSITY = 245
MIN_STD_INTENSITY = 5

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42


def validate_image_quality(filepath):
    """
    Validates chest X-ray image quality before inclusion in dataset.
    
    Args:
        filepath: Path to the image file
        
    Returns:
        tuple: (is_valid: bool, reason: str)
    """
    try:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return False, "Cannot read image"
        
        # Check resolution
        height, width = img.shape
        if height < MIN_RESOLUTION[0] or width < MIN_RESOLUTION[1]:
            return False, f"Resolution too low: {width}x{height}"
        
        # Check intensity range
        mean_intensity = np.mean(img)
        if mean_intensity < MIN_MEAN_INTENSITY:
            return False, "Image too dark (likely corrupted)"
        if mean_intensity > MAX_MEAN_INTENSITY:
            return False, "Image too bright (likely corrupted)"
        
        # Check contrast
        std_intensity = np.std(img)
        if std_intensity < MIN_STD_INTENSITY:
            return False, "Insufficient contrast (flat image)"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Error: {str(e)}"


def split_and_save_images(file_list, class_name):
    """
    Splits image list into train/val/test and saves with error tracking.
    
    Args:
        file_list: List of image file paths
        class_name: Class label ('NORMAL' or 'PNEUMONIA')
        
    Returns:
        tuple: (saved_counts: dict, failed_files: list)
    """
    # Split data
    train_files, temp_files = train_test_split(
        file_list, 
        test_size=(VAL_RATIO + TEST_RATIO), 
        random_state=RANDOM_SEED, 
        shuffle=True
    )
    val_files, test_files = train_test_split(
        temp_files, 
        test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)), 
        random_state=RANDOM_SEED, 
        shuffle=True
    )
    
    split_dict = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }
    
    saved_count = {"train": 0, "val": 0, "test": 0}
    failed_files = []
    
    # Save images to respective directories
    for split, files in split_dict.items():
        for filepath in tqdm(files, desc=f"Saving {split}/{class_name}"):
            img = cv2.imread(filepath)
            if img is None:
                failed_files.append((filepath, "Cannot read during save"))
                continue
            
            filename = os.path.basename(filepath)
            save_path = os.path.join(OUTPUT_DIR, split, class_name, filename)
            
            try:
                success = cv2.imwrite(save_path, img)
                if success:
                    saved_count[split] += 1
                else:
                    failed_files.append((filepath, "cv2.imwrite failed"))
            except Exception as e:
                failed_files.append((filepath, str(e)))
    
    return saved_count, failed_files


def main():
    """Main execution function for dataset organization."""
    
    print("="*70)
    print("CriticalCare-AI: Chest X-Ray Dataset Organization Pipeline")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize tracking
    quality_report = {
        "total_scanned": 0,
        "valid_images": 0,
        "rejected_images": 0,
        "rejection_reasons": {},
        "corrupted_files": []
    }
    
    
    print("STEP 1: Scanning and validating images...")
    print("-" * 70)
    
    images_by_class = {}
    
    for split in SPLITS:
        split_path = os.path.join(INPUT_DIR, split)
        for class_name in ["NORMAL", "PNEUMONIA"]:
            class_path = os.path.join(split_path, class_name)
            
            if not os.path.exists(class_path):
                print(f"⚠️  Warning: Path not found: {class_path}")
                continue
            
            valid_images = []
            files = [f for f in os.listdir(class_path) 
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            
            for f in tqdm(files, desc=f"Validating {split}/{class_name}"):
                filepath = os.path.join(class_path, f)
                quality_report["total_scanned"] += 1
                
                is_valid, reason = validate_image_quality(filepath)
                
                if is_valid:
                    valid_images.append(filepath)
                    quality_report["valid_images"] += 1
                else:
                    quality_report["rejected_images"] += 1
                    quality_report["rejection_reasons"][reason] = \
                        quality_report["rejection_reasons"].get(reason, 0) + 1
                    quality_report["corrupted_files"].append({
                        "file": filepath,
                        "reason": reason
                    })
            
            images_by_class[f"{split}_{class_name}"] = valid_images
    
    # Print quality validation report
    print("\n" + "="*70)
    print("IMAGE QUALITY VALIDATION REPORT")
    print("="*70)
    print(f"Total images scanned:  {quality_report['total_scanned']}")
    print(f"Valid images:          {quality_report['valid_images']}")
    print(f"Rejected images:       {quality_report['rejected_images']}")
    
    if quality_report['total_scanned'] > 0:
        rejection_rate = (quality_report['rejected_images'] / 
                         quality_report['total_scanned'] * 100)
        print(f"Rejection rate:        {rejection_rate:.2f}%")
    
    if quality_report["rejection_reasons"]:
        print("\nRejection reasons:")
        for reason, count in quality_report["rejection_reasons"].items():
            print(f"  • {reason}: {count}")
    
    
    print("\n" + "="*70)
    print("STEP 2: Organizing images by class...")
    print("-" * 70)
    
    all_normal_files = []
    all_pneumonia_files = []
    
    for key, files in images_by_class.items():
        if "NORMAL" in key:
            all_normal_files.extend(files)
        elif "PNEUMONIA" in key:
            all_pneumonia_files.extend(files)
    
    print(f"Total NORMAL images (validated):    {len(all_normal_files)}")
    print(f"Total PNEUMONIA images (validated): {len(all_pneumonia_files)}")
    print(f"Total images for training:          {len(all_normal_files) + len(all_pneumonia_files)}")
    
    # Calculate class imbalance
    total_images = len(all_normal_files) + len(all_pneumonia_files)
    if len(all_normal_files) > 0:
        imbalance_ratio = len(all_pneumonia_files) / len(all_normal_files)
        print(f"Class imbalance ratio (P/N):        {imbalance_ratio:.2f}:1")
    
    
    print("\n" + "="*70)
    print("STEP 3: Creating output directory structure...")
    print("-" * 70)
    
    for split in SPLITS:
        for label in ["NORMAL", "PNEUMONIA"]:
            dir_path = os.path.join(OUTPUT_DIR, split, label)
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ Created: {dir_path}")
    
    print("\n" + "="*70)
    print("STEP 4: Splitting and saving images (80/10/10 split)...")
    print("-" * 70)
    
    # Process NORMAL class
    normal_counts, normal_failed = split_and_save_images(all_normal_files, "NORMAL")
    
    # Process PNEUMONIA class
    pneumonia_counts, pneumonia_failed = split_and_save_images(all_pneumonia_files, "PNEUMONIA")
    
    # Report save failures
    total_failed = len(normal_failed) + len(pneumonia_failed)
    if total_failed > 0:
        print(f"\n⚠️  Warning: {total_failed} files failed to save")
        if total_failed <= 10:
            print("\nFailed files:")
            for filepath, reason in (normal_failed + pneumonia_failed):
                print(f"  • {os.path.basename(filepath)}: {reason}")
    
    
    print("\n" + "="*70)
    print("STEP 5: Verifying final class distribution...")
    print("="*70)
    
    dataset_metadata = {
        "dataset_name": "Chest X-Ray Pneumonia Detection - CriticalCare-AI",
        "source": "Kaggle - Paul Mooney Dataset",
        "preprocessing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "splits": {},
        "quality_metrics": quality_report,
        "preprocessing_parameters": {
            "min_resolution": MIN_RESOLUTION,
            "min_mean_intensity": MIN_MEAN_INTENSITY,
            "max_mean_intensity": MAX_MEAN_INTENSITY,
            "min_std_intensity": MIN_STD_INTENSITY,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "random_seed": RANDOM_SEED,
            "color_space": "grayscale",
            "target_size_for_training": "(224, 224)"
        }
    }
    
    for split in SPLITS:
        normal_path = os.path.join(OUTPUT_DIR, split, "NORMAL")
        pneumonia_path = os.path.join(OUTPUT_DIR, split, "PNEUMONIA")
        
        normal_count = len([f for f in os.listdir(normal_path)
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        pneumonia_count = len([f for f in os.listdir(pneumonia_path)
                              if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        
        total = normal_count + pneumonia_count
        normal_pct = (normal_count / total * 100) if total > 0 else 0
        pneumonia_pct = (pneumonia_count / total * 100) if total > 0 else 0
        imbalance = (pneumonia_count / normal_count) if normal_count > 0 else 0
        
        print(f"\n{split.upper()} SET:")
        print(f"  NORMAL:              {normal_count:5d} ({normal_pct:5.2f}%)")
        print(f"  PNEUMONIA:           {pneumonia_count:5d} ({pneumonia_pct:5.2f}%)")
        print(f"  Total:               {total:5d}")
        print(f"  Imbalance (P/N):     {imbalance:.2f}:1")
        
        dataset_metadata["splits"][split] = {
            "NORMAL": normal_count,
            "PNEUMONIA": pneumonia_count,
            "total": total,
            "imbalance_ratio": round(imbalance, 2),
            "normal_percentage": round(normal_pct, 2),
            "pneumonia_percentage": round(pneumonia_pct, 2)
        }
    
    print("\n" + "="*70)
    print("STEP 6: Saving metadata and reports...")
    print("-" * 70)
    
    # Save metadata JSON
    metadata_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(dataset_metadata, f, indent=4)
    print(f"✓ Metadata saved: {metadata_path}")
    
    # Save quality report
    quality_report_path = os.path.join(OUTPUT_DIR, "quality_report.json")
    with open(quality_report_path, 'w') as f:
        json.dump(quality_report, f, indent=4)
    print(f"✓ Quality report saved: {quality_report_path}")
    
    # Save README
    readme_path = os.path.join(OUTPUT_DIR, "README.txt")
    with open(readme_path, 'w') as f:
        f.write("CriticalCare-AI: Chest X-Ray Pneumonia Dataset\n")
        f.write("="*50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Dataset Structure:\n")
        f.write("  train/\n")
        f.write("    NORMAL/       - Training normal X-rays\n")
        f.write("    PNEUMONIA/    - Training pneumonia X-rays\n")
        f.write("  val/\n")
        f.write("    NORMAL/       - Validation normal X-rays\n")
        f.write("    PNEUMONIA/    - Validation pneumonia X-rays\n")
        f.write("  test/\n")
        f.write("    NORMAL/       - Test normal X-rays\n")
        f.write("    PNEUMONIA/    - Test pneumonia X-rays\n\n")
        f.write("Files:\n")
        f.write("  - dataset_metadata.json: Complete dataset statistics\n")
        f.write("  - quality_report.json: Image quality validation report\n")
        f.write("  - README.txt: This file\n\n")
        f.write(f"Total Images: {total_images}\n")
        f.write(f"Split Ratio: {TRAIN_RATIO*100:.0f}% train, {VAL_RATIO*100:.0f}% val, {TEST_RATIO*100:.0f}% test\n")
    print(f"✓ README saved: {readme_path}")
    
    print("\n" + "="*70)
    print("✅ DATASET ORGANIZATION COMPLETE!")
    print("="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

# ============================================================================
# EXECUTE MAIN
# ============================================================================

if __name__ == "__main__":
    main()
