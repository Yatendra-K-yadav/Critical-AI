import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

print("Converting NumPy arrays to PyTorch tensors...")

# Convert to PyTorch tensors with correct format
# For grayscale images (1 channel), we need to handle the channel dimension properly
x_train_tensor = torch.from_numpy(x_train).float().permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
y_train_tensor = torch.from_numpy(y_train).long()

x_val_tensor = torch.from_numpy(x_val).float().permute(0, 3, 1, 2)
y_val_tensor = torch.from_numpy(y_val).long()

x_test_tensor = torch.from_numpy(x_test).float().permute(0, 3, 1, 2)
y_test_tensor = torch.from_numpy(y_test).long()

# Verify shapes
print("\nTensor shapes:")
print(f"Train: {x_train_tensor.shape}, Labels: {y_train_tensor.shape}")
print(f"Val:   {x_val_tensor.shape}, Labels: {y_val_tensor.shape}")
print(f"Test:  {x_test_tensor.shape}, Labels: {y_test_tensor.shape}")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Move tensors to GPU if available (optional but recommended for speed)
if torch.cuda.is_available():
    x_train_tensor = x_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    x_val_tensor = x_val_tensor.to(device)
    y_val_tensor = y_val_tensor.to(device)
    x_test_tensor = x_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)
    print("✓ Tensors moved to GPU")
else:
    print("⚠ GPU not available, using CPU (training will be slower)")

print("\n✅ Tensor conversion complete!")
