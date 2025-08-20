# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import scipy.linalg
from . import metric_utils

#----------------------------------------------------------------------------
"""ORIGNAL """
def compute_fid(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    if opts.rank != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

#----------------------------------------------------------------------------
"""NEW"""
def compute_fid_en(opts_hr, opts_lr, max_real, num_gen):
    """
    Args:
        opts_hr: Options with HR dataset for real images
        opts_lr: Options with LR dataset for generator inputs
        max_real: Maximum number of real HR images
        num_gen: Maximum number of generated images
    """
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.


    # Compute features for real HR images
    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts_hr, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

    # print(">>After : compute_feature_stats_for_dataset")

    # Compute features for generated HR images (from LR inputs)
    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
        opts=opts_lr, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    # print(">>After : compute_feature_stats_for_generator")

    if opts_hr.rank != 0:
        return float('nan')
    
    eps = 1e-6
    sigma_real += np.eye(sigma_real.shape[0]) * eps
    sigma_gen  += np.eye(sigma_gen.shape[0]) * eps


    covmean, info = scipy.linalg.sqrtm(sigma_gen @ sigma_real, disp=False)

    # If result has tiny imaginary components, discard them
    if np.iscomplexobj(covmean):
        covmean = covmean.real


    m = np.sum((mu_gen - mu_real)**2)
    fid = m + np.trace(sigma_gen + sigma_real - 2*covmean)


    # m = np.square(mu_gen - mu_real).sum()
    # s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    # fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

    
    
    
