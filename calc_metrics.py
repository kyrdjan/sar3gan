# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Calculate quality metrics for previous training run or pretrained network pickle."""

import os
import click
import json
import tempfile
import copy
import torch

import dnnlib
import legacy
from metrics import metric_main
from metrics import metric_utils
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0 or not args.verbose:
        custom_ops.verbosity = 'none'

    # Configure torch.
    device = torch.device('cuda', rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True

    # Print network summary.
    G = copy.deepcopy(args.G).eval().requires_grad_(False).to(device)
    if rank == 0 and args.verbose:
        print("Printing G's network summary tables...")
        try:
            # Load a sample from the dataset to get the shape
            training_set = dnnlib.util.construct_class_by_name(**args.G_training_set_kwargs)
            lr_shape = training_set[0][0].shape
            lr_img = torch.empty([1, *lr_shape], device=device)
            c = torch.empty([1, G.c_dim], device=device)
            misc.print_module_summary(G, [lr_img, c])
        except Exception as e:
            print(f"Could not print network summary: {e}")
            print("Attempting with default shape...")
            try:
                # Fallback: use img_resolution from G
                lr_img = torch.empty([1, 3, G.img_resolution, G.img_resolution], device=device)
                c = torch.empty([1, G.c_dim], device=device)
                misc.print_module_summary(G, [lr_img, c])
            except Exception as e2:
                print(f"Network summary failed: {e2}")

    # Calculate each metric.
    for metric in args.metrics:
        if rank == 0 and args.verbose:
            print(f'Calculating {metric}...')
        progress = metric_utils.ProgressMonitor(verbose=args.verbose)
        result_dict = metric_main.calc_metric(
            metric=metric, 
            G=G, 
            G_dataset_kwargs=args.G_training_set_kwargs,
            D_dataset_kwargs=args.D_training_set_kwargs, 
            num_gpus=args.num_gpus, 
            rank=rank, 
            device=device, 
            progress=progress
        )
        if rank == 0:
            metric_main.report_metric(result_dict, run_dir=args.run_dir, snapshot_pkl=args.network_pkl)
        if rank == 0 and args.verbose:
            print()

    # Done.
    if rank == 0 and args.verbose:
        print('Exiting...')
#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('network_pkl', '--network', help='Network pickle filename or URL', metavar='PATH', required=True)
@click.option('--metrics', help='Quality metrics', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--data_lr', help='Low-res dataset path', metavar='[ZIP|DIR]', required=True)
@click.option('--data_hr', help='High-res dataset path', metavar='[ZIP|DIR]', required=True)
@click.option('--mirror', help='Enable dataset x-flips  [default: look up]', type=bool, metavar='BOOL')
@click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT', show_default=True)
@click.option('--verbose', help='Print optional information', type=bool, default=True, metavar='BOOL', show_default=True)

def calc_metrics(ctx, network_pkl, metrics, data_lr, data_hr, mirror, gpus, verbose):
    """Calculate quality metrics for previous training run or pretrained network pickle.

    Examples:

    \b
    # Previous training run: look up options automatically, save result to JSONL file.
    (no use) python calc_metrics.py --metrics=eqt50k_int,eqr50k \\
        --network=~/training-runs/00000-stylegan3-r-mydataset/network-snapshot-000000.pkl

    \b
    # Pre-trained network pickle: specify dataset explicitly, print result to stdout.
    (no use) python calc_metrics.py --metrics=fid50k_full --data=~/datasets/ffhq-1024x1024.zip --mirror=1 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl

    \b    
    # For FID use data_hr:
      python calc_metrics.py --metrics=fid50k --data_hr=./datasets/ffhq-1024x1024.zip --network=network-snapshot-000100.pkl

    \b
    # For PSNR and SSIM:

    \b
    thesis metrics:
      ssim         Structural similarity index measure against 50k real images.
      fid_en       Frechet inception distance against 10k real images.
      psnr_en      Peak signal noise ratio against 50k real images.

    \b
    Recommended metrics:
      fid50k_full  Frechet inception distance against the full dataset.
      kid50k_full  Kernel inception distance against the full dataset.
      pr50k3_full  Precision and recall againt the full dataset.

    \b
    Legacy metrics:
      fid50k       Frechet inception distance against 50k real images.
      kid50k       Kernel inception distance against 50k real images.
      pr50k3       Precision and recall against 50k real images.
      is50k        Inception score for CIFAR-10.
    """
    dnnlib.util.Logger(should_flush=True)

    # Validate arguments.
    args = dnnlib.EasyDict(metrics=metrics, num_gpus=gpus, network_pkl=network_pkl, verbose=verbose)
    if not all(metric_main.is_valid_metric(metric) for metric in args.metrics):
        ctx.fail('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    if not args.num_gpus >= 1:
        ctx.fail('--gpus must be at least 1')

    # Validate dataset paths
    if data_lr is None or data_hr is None:
        ctx.fail('Both --data_lr and --data_hr must be specified')
    
    if not os.path.exists(data_lr):
        ctx.fail(f'Low-res dataset path does not exist: {data_lr}')
    
    if not os.path.exists(data_hr):
        ctx.fail(f'High-res dataset path does not exist: {data_hr}')

    # Load network.
    if not dnnlib.util.is_url(network_pkl, allow_file_urls=True) and not os.path.isfile(network_pkl):
        ctx.fail('--network must point to a file or URL')
    if args.verbose:
        print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=args.verbose) as f:
        network_dict = legacy.load_network_pkl(f)
        args.G = network_dict['G_ema'] # subclass of torch.nn.Module

    # Initialize dataset options.
    args.G_training_set_kwargs = dnnlib.EasyDict(
        class_name='training.dataset.ImageFolderDataset',
        path=data_lr
    )
    args.D_training_set_kwargs = dnnlib.EasyDict(
        class_name='training.dataset.ImageFolderDataset',
        path=data_hr
    )
    
    if args.verbose:
        print("LR dataset stored in G_training_set_kwargs")
        print("HR dataset stored in D_training_set_kwargs")

    # Finalize dataset options for both G and D datasets.
    args.G_training_set_kwargs.resolution = args.G.img_resolution
    args.G_training_set_kwargs.use_labels = (args.G.c_dim != 0)
    args.D_training_set_kwargs.resolution = args.G.img_resolution
    args.D_training_set_kwargs.use_labels = (args.G.c_dim != 0)
    
    if mirror is not None:
        args.G_training_set_kwargs.xflip = mirror
        args.D_training_set_kwargs.xflip = mirror

    # Print dataset options.
    if args.verbose:
        print('LR Dataset options (G):')
        print(json.dumps(args.G_training_set_kwargs, indent=2))
        print('HR Dataset options (D):')
        print(json.dumps(args.D_training_set_kwargs, indent=2))

    # Locate run dir.
    args.run_dir = None
    if os.path.isfile(network_pkl):
        pkl_dir = os.path.dirname(network_pkl)
        if os.path.isfile(os.path.join(pkl_dir, 'training_options.json')):
            args.run_dir = pkl_dir

    # Launch processes.
    if args.verbose:
        print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)
    
    
    # dnnlib.util.Logger(should_flush=True)

    # # Validate arguments.
    # args = dnnlib.EasyDict(metrics=metrics, num_gpus=gpus, network_pkl=network_pkl, verbose=verbose)
    # if not all(metric_main.is_valid_metric(metric) for metric in args.metrics):
    #     ctx.fail('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    # if not args.num_gpus >= 1:
    #     ctx.fail('--gpus must be at least 1')

    # # Load network.
    # if not dnnlib.util.is_url(network_pkl, allow_file_urls=True) and not os.path.isfile(network_pkl):
    #     ctx.fail('--network must point to a file or URL')
    # if args.verbose:
    #     print(f'Loading network from "{network_pkl}"...')
    # with dnnlib.util.open_url(network_pkl, verbose=args.verbose) as f:
    #     network_dict = legacy.load_network_pkl(f)
    #     args.G = network_dict['G_ema'] # subclass of torch.nn.Module

    # # Initialize dataset options. # dirty
    # # if data is not None: # old
    # #     args.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data)
    # # if data_lr is not None and data_hr is not None: # new
    # #     args.G_training_set_kwargs = dnnlib.EasyDict(
    # #         class_name='training.dataset.ImageFolderDataset',
    # #         path=data_lr
    # #     )
    # #     args.D_training_set_kwargs = dnnlib.EasyDict(
    # #         class_name='training.dataset.ImageFolderDataset',
    # #         path=data_hr
    # #     )
    # #     print("lr and hr are stored in G_training_set_kwargs and D_training_set_kwargs!")
    # # # elif data_hr is not None: 
    # # #     args.dataset_kwargs = dnnlib.EasyDict(
    # # #         class_name='training.dataset.ImageFolderDataset',
    # # #         path=data_hr
    # # #     )
    # # # elif network_dict['training_set_kwargs'] is not None:
    # # #     args.dataset_kwargs = dnnlib.EasyDict(network_dict['training_set_kwargs'])
    # # else:
    # #     ctx.fail('Could not look up dataset options; please specify --data_lr/--data_hr')


    # # Initialize dataset options. # clean
    # if data_lr is not None and data_hr is not None: # new
    #     args.G_training_set_kwargs = dnnlib.EasyDict(
    #         class_name='training.dataset.ImageFolderDataset',
    #         path=data_lr
    #     )
    #     args.D_training_set_kwargs = dnnlib.EasyDict(
    #         class_name='training.dataset.ImageFolderDataset',
    #         path=data_hr
    #     )
    #     print("lr and hr are stored in G_training_set_kwargs and D_training_set_kwargs!")
    # else:
    #     ctx.fail('Could not look up dataset options; please specify --data_lr/--data_hr')

    # # Finalize dataset options.
    # args.G_training_set_kwargs.resolution = args.G.img_resolution
    # args.G_training_set_kwargs.use_labels = (args.G.c_dim != 0)
    # if mirror is not None:
    #     args.G_training_set_kwargs.xflip = mirror

    # # Print dataset options.
    # if args.verbose:
    #     print('Dataset options:')
    #     print(json.dumps(args.G_training_set_kwargs, indent=2))
    #     print(json.dumps(args.D_training_set_kwargs, indent=2))

    # # Locate run dir.
    # args.run_dir = None
    # if os.path.isfile(network_pkl):
    #     pkl_dir = os.path.dirname(network_pkl)
    #     if os.path.isfile(os.path.join(pkl_dir, 'training_options.json')):
    #         args.run_dir = pkl_dir

    # # Launch processes.
    # if args.verbose:
    #     print('Launching processes...')
    # torch.multiprocessing.set_start_method('spawn')
    # with tempfile.TemporaryDirectory() as temp_dir:
    #     if args.num_gpus == 1:
    #         subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
    #     else:
    #         torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    calc_metrics() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
