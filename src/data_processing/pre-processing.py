"""
CriticalCare-AI: Enhanced Medical Image Preprocessing Pipeline
Robust preprocessing for chest X-ray pneumonia detection
Author: CriticalCare-AI Team
Date: October 2025
"""

import os
import cv2
import numpy as np
from collections import Counter
from tqdm import tqdm
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT_DIR = r"C:\Sem 7\LY Project\Project\Critical-AI\chest_xray_split"
SPLITS = ["train", "val", "test"]
LABEL_MAP = {"NORMAL": 0, "PNEUMONIA": 1}

# Enhanced preprocessing parameters
TARGET_SIZE = (224, 224)
NORMALIZE = True
APPLY_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# ============================================================================
# NUMPY TO PYTHON TYPE CONVERTER (FIX FOR JSON SERIALIZATION)
# ============================================================================

class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# ============================================================================
# ENHANCED PREPROCESSING FUNCTIONS
# ============================================================================

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def compute_dataset_statistics(images):
    """Compute mean and standard deviation for the dataset."""
    pixel_sum = 0
    pixel_sq_sum = 0
    total_pixels = 0
    
    for img in images:
        pixel_sum += np.sum(img)
        pixel_sq_sum += np.sum(img ** 2)
        total_pixels += img.size
    
    mean = pixel_sum / total_pixels
    std = np.sqrt(pixel_sq_sum / total_pixels - mean ** 2)
    
    return mean, std


def preprocess_image(image, target_size=(224, 224), normalize=True, 
                    apply_clahe_flag=True, mean=None, std=None):
    """Complete preprocessing pipeline for chest X-ray images."""
    # Step 1: Apply CLAHE for contrast enhancement
    if apply_clahe_flag:
        image = apply_clahe(image, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE)
    
    # Step 2: Resize to target dimensions
    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Step 3: Convert to float and normalize
    image = image.astype("float32")
    
    if normalize:
        if mean is not None and std is not None:
            # Z-score normalization using dataset statistics
            image = (image - mean) / (std + 1e-8)
        else:
            # Simple 0-1 normalization
            image = image / 255.0
    
    return image


def validate_image_during_load(image, min_resolution=(400, 400)):
    """Validate image quality during loading phase."""
    if image is None:
        return False, "Cannot read image"
    
    height, width = image.shape
    if height < min_resolution[0] or width < min_resolution[1]:
        return False, f"Resolution too low: {width}x{height}"
    
    if np.std(image) < 5:
        return False, "Insufficient contrast (likely corrupted)"
    
    return True, "Valid"


# ============================================================================
# ENHANCED LOADING FUNCTION
# ============================================================================

def load_split(split_name, target_size=(224, 224), normalize=True, 
               apply_clahe_flag=True, compute_stats=False):
    """Enhanced loading function with medical imaging best practices."""
    images = []
    labels = []
    bad_files = []
    rejected_files = []
    
    split_path = os.path.join(ROOT_DIR, split_name)
    
    print(f"\nLoading and preprocessing {split_name} set...")
    print("-" * 70)
    
    # First pass: Load raw images for statistics computation if needed
    raw_images = []
    
    for class_name, class_index in LABEL_MAP.items():
        class_path = os.path.join(split_path, class_name)
        
        if not os.path.exists(class_path):
            print(f"⚠️  Warning: Path not found: {class_path}")
            continue
        
        files = [f for f in os.listdir(class_path) 
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        for filename in tqdm(files, desc=f"Processing {split_name}/{class_name}"):
            image_path = os.path.join(class_path, filename)
            
            # Read image in grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                bad_files.append((image_path, "Cannot read"))
                continue
            
            # Validate image quality
            is_valid, reason = validate_image_during_load(img)
            if not is_valid:
                rejected_files.append((image_path, reason))
                continue
            
            # Store raw image for statistics if needed
            if compute_stats:
                raw_images.append(img.copy())
            
            images.append(img)
            labels.append(class_index)
    
    # Compute dataset statistics if requested
    mean, std = None, None
    if compute_stats and len(raw_images) > 0:
        print("\nComputing dataset statistics...")
        mean, std = compute_dataset_statistics(raw_images)
        print(f"Dataset mean: {mean:.4f}")
        print(f"Dataset std:  {std:.4f}")
    
    # Second pass: Apply preprocessing
    print(f"\nApplying preprocessing pipeline...")
    processed_images = []
    
    for img in tqdm(images, desc="Preprocessing"):
        processed_img = preprocess_image(
            img, 
            target_size=target_size, 
            normalize=normalize,
            apply_clahe_flag=apply_clahe_flag,
            mean=mean,
            std=std
        )
        
        # Expand channel dimension to (H, W, 1)
        processed_img = np.expand_dims(processed_img, axis=-1)
        processed_images.append(processed_img)
    
    # Report issues
    if bad_files:
        print(f"\n⚠️  Warning: Skipped {len(bad_files)} corrupted files")
    
    if rejected_files:
        print(f"⚠️  Warning: Rejected {len(rejected_files)} low-quality images")
    
    # Save preprocessing report with NumPy type conversion
    preprocessing_report = {
        "split": split_name,
        "total_loaded": len(processed_images),
        "corrupted_files": len(bad_files),
        "rejected_files": len(rejected_files),
        "preprocessing_params": {
            "target_size": target_size,
            "normalize": normalize,
            "apply_clahe": apply_clahe_flag,
            "clahe_clip_limit": CLAHE_CLIP_LIMIT if apply_clahe_flag else None,
            "clahe_tile_grid_size": CLAHE_TILE_GRID_SIZE if apply_clahe_flag else None,
            "dataset_mean": float(mean) if mean is not None else None,
            "dataset_std": float(std) if std is not None else None
        },
        "bad_files": [{"file": f, "reason": r} for f, r in bad_files[:10]],
        "rejected_files": [{"file": f, "reason": r} for f, r in rejected_files[:10]]
    }
    
    report_path = os.path.join(ROOT_DIR, f"{split_name}_preprocessing_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(preprocessing_report, f, indent=4, cls=NpEncoder)
    
    print(f"✓ Preprocessing report saved: {report_path}")
    
    if compute_stats:
        return processed_images, labels, mean, std
    else:
        return processed_images, labels


# ============================================================================
# MAIN LOADING PIPELINE
# ============================================================================

def main():
    """Main execution for loading and preprocessing all splits."""
    
    print("="*70)
    print("CriticalCare-AI: Enhanced Medical Image Loading Pipeline")
    print("="*70)
    
    # Load training set and compute statistics
    x_train, y_train, train_mean, train_std = load_split(
        "train", 
        target_size=TARGET_SIZE,
        normalize=NORMALIZE,
        apply_clahe_flag=APPLY_CLAHE,
        compute_stats=True
    )
    
    # Load validation set
    print(f"\nUsing training statistics for validation set normalization...")
    x_val, y_val = load_split(
        "val",
        target_size=TARGET_SIZE,
        normalize=NORMALIZE,
        apply_clahe_flag=APPLY_CLAHE,
        compute_stats=False
    )
    
    # Load test set
    print(f"\nUsing training statistics for test set normalization...")
    x_test, y_test = load_split(
        "test",
        target_size=TARGET_SIZE,
        normalize=NORMALIZE,
        apply_clahe_flag=APPLY_CLAHE,
        compute_stats=False
    )
    
    # Convert lists to NumPy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    # Display dataset statistics
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Train set:      {x_train.shape}, labels: {y_train.shape}")
    print(f"Validation set: {x_val.shape}, labels: {y_val.shape}")
    print(f"Test set:       {x_test.shape}, labels: {y_test.shape}")
    
    # Show class balance
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION")
    print("="*70)
    
    train_balance = Counter(y_train)
    val_balance = Counter(y_val)
    test_balance = Counter(y_test)
    
    # Convert numpy int64 to Python int for display
    print(f"Train:      NORMAL={int(train_balance[0])}, PNEUMONIA={int(train_balance[1])}")
    print(f"Validation: NORMAL={int(val_balance[0])}, PNEUMONIA={int(val_balance[1])}")
    print(f"Test:       NORMAL={int(test_balance[0])}, PNEUMONIA={int(test_balance[1])}")
    
    # Calculate and display class imbalance ratios
    train_ratio = train_balance[1] / train_balance[0] if train_balance[0] > 0 else 0
    val_ratio = val_balance[1] / val_balance[0] if val_balance[0] > 0 else 0
    test_ratio = test_balance[1] / test_balance[0] if test_balance[0] > 0 else 0
    
    print("\n" + "="*70)
    print("CLASS IMBALANCE RATIOS (PNEUMONIA/NORMAL)")
    print("="*70)
    print(f"Train:      {train_ratio:.2f}:1")
    print(f"Validation: {val_ratio:.2f}:1")
    print(f"Test:       {test_ratio:.2f}:1")
    
    # Recommendation for handling imbalance
    if train_ratio > 1.5:
        print("\n⚠️  High class imbalance detected!")
        print("Recommendations:")
        print("  1. Use class weights during training")
        print("  2. Consider data augmentation for minority class")
        print("  3. Use appropriate evaluation metrics (ROC-AUC, F1-score)")
    
    # Save preprocessing metadata with proper type conversion
    metadata = {
        "preprocessing_date": "2025-10-05",
        "preprocessing_params": {
            "target_size": list(TARGET_SIZE),
            "normalization": "dataset-specific Z-score" if NORMALIZE else "none",
            "clahe_applied": APPLY_CLAHE,
            "clahe_clip_limit": float(CLAHE_CLIP_LIMIT),
            "clahe_tile_grid_size": list(CLAHE_TILE_GRID_SIZE)
        },
        "dataset_statistics": {
            "train_mean": float(train_mean),
            "train_std": float(train_std)
        },
        "shapes": {
            "x_train": [int(s) for s in x_train.shape],
            "x_val": [int(s) for s in x_val.shape],
            "x_test": [int(s) for s in x_test.shape]
        },
        "class_distribution": {
            "train": {"NORMAL": int(train_balance[0]), "PNEUMONIA": int(train_balance[1])},
            "val": {"NORMAL": int(val_balance[0]), "PNEUMONIA": int(val_balance[1])},
            "test": {"NORMAL": int(test_balance[0]), "PNEUMONIA": int(test_balance[1])}
        }
    }
    
    metadata_path = os.path.join(ROOT_DIR, "preprocessing_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, cls=NpEncoder)
    
    print(f"\n✅ Preprocessing metadata saved: {metadata_path}")
    print("="*70)
    
    return x_train, y_train, x_val, y_val, x_test, y_test


# ============================================================================
# EXECUTE
# ============================================================================

if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, y_test = main()
