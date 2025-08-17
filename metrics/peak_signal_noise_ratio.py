
import numpy as np
import torch
import copy
import dnnlib
from torch.utils.data import DataLoader
import torch.nn.functional as F


def compute_psnr_enhanced(opts_hr, opts_lr, max_real=None, num_gen=50000):
    """
    Enhanced PSNR computation for separate LR and HR datasets.
    
    Args:
        opts_hr: MetricOptions for HR (ground truth) dataset
        opts_lr: MetricOptions for LR (input) dataset  
        max_real: Maximum number of real images to use
        num_gen: Number of images to generate
    
    Returns:
        dict: PSNR statistics
    """
    
    # Clone generator from LR options
    G = copy.deepcopy(opts_lr.G).eval().requires_grad_(False).to(opts_lr.device)
    
    # Build datasets
    lr_dataset = dnnlib.util.construct_class_by_name(**opts_lr.dataset_kwargs)
    hr_dataset = dnnlib.util.construct_class_by_name(**opts_hr.dataset_kwargs)
    
    # Determine how many images to process
    dataset_length = min(len(lr_dataset), len(hr_dataset))
    if max_real is not None:
        dataset_length = min(dataset_length, max_real)
    if num_gen is not None:
        dataset_length = min(dataset_length, num_gen)
    
    # Create synchronized data loaders
    batch_size = 64
    indices = list(range(dataset_length))
    
    lr_loader = DataLoader(lr_dataset, batch_size=batch_size, shuffle=False,
                          sampler=indices, num_workers=0, drop_last=False)
    hr_loader = DataLoader(hr_dataset, batch_size=batch_size, shuffle=False,
                          sampler=indices, num_workers=0, drop_last=False)
    
    psnr_values = []
    processed = 0
    
    progress = opts_lr.progress.sub(tag='PSNR computation', num_items=dataset_length)
    
    # Process paired batches
    lr_iter = iter(lr_loader)
    hr_iter = iter(hr_loader)
    
    while processed < dataset_length:
        try:
            lr_data = next(lr_iter)
            hr_data = next(hr_iter)
        except StopIteration:
            break
            
        # Extract images (handle both (image, label) and (lr, hr) formats)
        lr_batch = lr_data[0] if isinstance(lr_data, (list, tuple)) else lr_data
        hr_batch = hr_data[0] if isinstance(hr_data, (list, tuple)) else hr_data
        
        # Ensure we don't exceed our target count
        current_batch_size = min(lr_batch.shape[0], dataset_length - processed)
        lr_batch = lr_batch[:current_batch_size]
        hr_batch = hr_batch[:current_batch_size]
        
        # Move to device
        lr_batch = lr_batch.to(opts_lr.device).to(torch.float32)
        hr_batch = hr_batch.to(opts_lr.device).to(torch.float32)
        
        # Generate enhanced images
        with torch.no_grad():
            if G.c_dim > 0:
                c = torch.zeros([lr_batch.shape[0], G.c_dim], device=opts_lr.device)
                generated_hr = G(lr_batch, c)
            else:
                generated_hr = G(lr_batch)
        
        # Normalize both images to [0, 255] range
        generated_hr = normalize_images(generated_hr)
        hr_batch = normalize_images(hr_batch)
        # TEMPO
        if generated_hr.shape[-2:] != hr_batch.shape[-2:]:
            generated_hr = F.interpolate(
                generated_hr, size=hr_batch.shape[-2:], 
                mode='bilinear', align_corners=False
            )

        
        # Calculate PSNR for this batch
        batch_psnr = calculate_psnr_batch(generated_hr, hr_batch)
        psnr_values.extend(batch_psnr)
        
        processed += current_batch_size
        progress.update(processed)
    
    # Calculate final statistics
    psnr_array = np.array(psnr_values[:dataset_length])
    
    # Filter out infinite values for statistics (perfect matches)
    finite_psnr = psnr_array[np.isfinite(psnr_array)]
    
    results = {
        'psnr_mean': float(np.mean(finite_psnr)) if len(finite_psnr) > 0 else 0.0,
        'psnr_std': float(np.std(finite_psnr)) if len(finite_psnr) > 0 else 0.0,
        'psnr_median': float(np.median(finite_psnr)) if len(finite_psnr) > 0 else 0.0,
        'psnr_min': float(np.min(finite_psnr)) if len(finite_psnr) > 0 else 0.0,
        'psnr_max': float(np.max(finite_psnr)) if len(finite_psnr) > 0 else 0.0,
        'num_perfect_matches': int(np.sum(np.isinf(psnr_array))),
        'num_images': len(psnr_array)
    }
    
    return results


def normalize_images(images):
    """
    Normalize images to [0, 255] range for PSNR calculation.
    
    Args:
        images: Tensor of images
    
    Returns:
        torch.Tensor: Normalized images in [0, 255] range
    """
    if images.min() < 0:
        # Assume [-1, 1] range
        images = (images + 1) * 127.5
    elif images.max() <= 1.0:
        # Assume [0, 1] range  
        images = images * 255.0
    # Otherwise assume already in [0, 255] range
    
    return torch.clamp(images, 0, 255)


def calculate_psnr_batch(img1, img2, max_val=255.0):
    """
    Calculate PSNR for a batch of images.
    
    Args:
        img1: Generated images tensor [N, C, H, W]
        img2: Ground truth images tensor [N, C, H, W]  
        max_val: Maximum pixel value (default 255.0)
    
    Returns:
        list: PSNR values for each image in the batch
    """
    batch_size = img1.shape[0]
    psnr_values = []
    
    for i in range(batch_size):
        # Calculate MSE for this image
        mse = torch.mean((img1[i] - img2[i]) ** 2)
        
        # Avoid division by zero
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
            psnr = float(psnr.cpu())
        
        psnr_values.append(psnr)
    
    return psnr_values

def calculate_psnr_numpy(img1, img2, max_val=255.0):
    """
    Calculate PSNR between two numpy arrays.
    
    Args:
        img1: First image array
        img2: Second image array
        max_val: Maximum pixel value
    
    Returns:
        float: PSNR value
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return float(psnr)