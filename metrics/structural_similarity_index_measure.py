import numpy as np
import torch
import torch.nn.functional as F
import dnnlib
from . import metric_utils

def create_gaussian_window(window_size, sigma, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    return window.unsqueeze(0).unsqueeze(0)

def compute_ssim_batch(img1, img2, window_size=11, sigma=1.5):
    """
    Compute SSIM between two batches of images.
    
    Args:
        img1, img2: Tensors of shape [B, C, H, W] with values in [0, 255]
        window_size: Size of the Gaussian window
        sigma: Standard deviation for Gaussian window
    Returns:
        SSIM values for each image in the batch
    """  
    # Normalize to [0, 1]
    img1 = img1.float() / 255.0
    img2 = img2.float() / 255.0
    
    # Create Gaussian Window
    window = create_gaussian_window(window_size, sigma, img1.device)
    C = img1.size(1)
    window = window.expand(C, 1, window_size, window_size).contiguous()
    
    # SSIM constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Compute local means 
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=C)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Compute local variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=C) - mu1_mu2
    
    # Compute SSIM map
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # Return mean SSIM for each image in batch
    return ssim_map.view(ssim_map.size(0), -1).mean(dim=1)

def compute_ssim(opts, num_gen, max_real=None):
    """
    Compute SSIM between generated and real images.
    
    Args:
        opts: MetricOptions containing generator and dataset info
        num_gen: Number of images to generate for evaluation
        max_real: Maximum number of real images to use (for compatibility)
    
    Returns:
        Mean and standard deviation of SSIM scores
    """
    
    # Skip feature extraction and go directly to SSIM computation
    if opts.rank != 0:
        return float('nan'), float('nan')
        
    return compute_ssim_direct(opts, num_gen, max_real)

def compute_ssim_direct(opts, num_gen, max_real):
    """Direct SSIM Computation without using feature extraction"""
    import copy
    from torch.utils.data import DataLoader
    
    # Create a copy of the generator for evaluation
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    
    # Build dataset - this should be your HR dataset for comparison
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    
    batch_size = min(32, num_gen)  # Use smaller batch size to avoid memory issues
    num_batches = min((num_gen + batch_size - 1) // batch_size, len(dataset) // batch_size)
    
    ssim_scores = []
    processed = 0

    # Create progress tracker if available
    progress = opts.progress.sub(tag='SSIM computation', num_items=num_gen) if hasattr(opts, 'progress') else None
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    for batch_idx, batch_data in enumerate(dataloader):
        if batch_idx >= num_batches or processed >= num_gen:
            break
        
        # Handle different dataset formats
        if isinstance(batch_data, (list, tuple)):
            if len(batch_data) >= 2:
                # Format: (lr_images, hr_images, [labels])
                lr_images = batch_data[0].to(opts.device)
                hr_real = batch_data[1].to(opts.device)
                labels = batch_data[2] if len(batch_data) > 2 else None
            else:
                # Format: (images, [labels]) - assume this is HR, generate LR by downsampling
                hr_real = batch_data[0].to(opts.device)
                # Simple downsampling for LR (you might want to use your actual LR generation method)
                lr_images = F.interpolate(hr_real, scale_factor=0.25, mode='bilinear', align_corners=False)
                lr_images = F.interpolate(lr_images, size=hr_real.shape[-2:], mode='bilinear', align_corners=False)
                labels = batch_data[1] if len(batch_data) > 1 else None
        else:
            # Single tensor - assume HR
            hr_real = batch_data.to(opts.device)
            lr_images = F.interpolate(hr_real, scale_factor=0.25, mode='bilinear', align_corners=False)
            lr_images = F.interpolate(lr_images, size=hr_real.shape[-2:], mode='bilinear', align_corners=False)
            labels = None
        
        # Handle conditional generation
        if G.c_dim > 0:
            if labels is not None:
                c = labels.to(opts.device)
            else:
                c = torch.zeros([lr_images.shape[0], G.c_dim], device=opts.device)
        else:
            c = None
            
        # Generate HR images
        with torch.no_grad():
            if c is not None:
                hr_generated = G(lr_images, c)
            else:
                hr_generated = G(lr_images)
        
        # Convert from [-1, 1] to [0, 255] range (assuming your GAN outputs in [-1, 1])
        hr_real_uint8 = (hr_real * 127.5 + 127.5).clamp(0, 255)
        hr_gen_uint8 = (hr_generated * 127.5 + 127.5).clamp(0, 255)
        
        # Compute SSIM
        try:
            batch_ssim = compute_ssim_batch(hr_gen_uint8, hr_real_uint8)
            ssim_scores.extend(batch_ssim.cpu().numpy())
        except Exception as e:
            print(f"Error computing SSIM for batch {batch_idx}: {e}")
            # Add NaN values for failed batches
            ssim_scores.extend([float('nan')] * lr_images.shape[0])
        
        processed += lr_images.shape[0]
        if progress is not None:
            progress.update(min(processed, num_gen))
    
    # Convert to numpy and filter out NaN values
    ssim_scores = np.array(ssim_scores[:num_gen])
    valid_scores = ssim_scores[~np.isnan(ssim_scores)]
    
    if len(valid_scores) == 0:
        print("Warning: No valid SSIM scores computed")
        return float('nan'), float('nan')
    
    mean_ssim = float(np.mean(valid_scores))
    std_ssim = float(np.std(valid_scores))
    
    print(f"SSIM computed on {len(valid_scores)}/{len(ssim_scores)} samples")
    print(f"Mean SSIM: {mean_ssim:.4f}, Std SSIM: {std_ssim:.4f}")
    
    return mean_ssim, std_ssim