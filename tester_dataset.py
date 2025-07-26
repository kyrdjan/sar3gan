import torch
from torchvision.utils import save_image
import os
from training.dataset import PairedImageFolderDataset

# Initialize the dataset using the correct folder paths inside the zip
dataset = PairedImageFolderDataset(
    path=r'datasets/Super Resolution-ImageNet-Pairs.zip',  # Use raw string or forward slashes
    resolution=None,
    low_dir='super resolution/low_resolution',    # Folder path inside the zip
    high_dir='super resolution/high_resolution'   # Folder path inside the zip
)

print(f"Dataset length: {len(dataset)}")

# Create a folder to save the samples
os.makedirs('test_output', exist_ok=True)

# Sample 3 paired images and save them
for idx in range(3):
    low, high = dataset[idx]  # Numpy arrays in CHW format

    # Convert to PyTorch tensors and normalize to [0, 1]
    low_tensor = torch.from_numpy(low).float() / 255.0
    high_tensor = torch.from_numpy(high).float() / 255.0

    # Save images
    save_image(low_tensor, os.path.join('test_output', f'low_{idx}.png'))
    save_image(high_tensor, os.path.join('test_output', f'high_{idx}.png'))

    print(f"low and high res Saved pair {idx}")
