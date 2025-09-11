# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main

def cosine_decay_with_warmup(cur_nimg, base_value, total_nimg, final_value=0.0, warmup_value=0.0, warmup_nimg=0, hold_base_value_nimg=0):
    decay = 0.5 * (1 + np.cos(np.pi * (cur_nimg - warmup_nimg - hold_base_value_nimg) / float(total_nimg - warmup_nimg - hold_base_value_nimg)))
    cur_value = base_value + (1 - decay) * (final_value - base_value)
    if hold_base_value_nimg > 0:
        cur_value = np.where(cur_nimg > warmup_nimg + hold_base_value_nimg, cur_value, base_value)
    if warmup_nimg > 0:
        slope = (base_value - warmup_value) / warmup_nimg
        warmup_v = slope * cur_nimg + warmup_value
        cur_value = np.where(cur_nimg < warmup_nimg, warmup_v, cur_value)
    return float(np.where(cur_nimg > total_nimg, final_value, cur_value))

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def remap_optimizer_state_dict(state_dict, device):
    state_dict = copy.deepcopy(state_dict)
    for param in state_dict['state'].values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
    return state_dict

#----------------------------------------------------------------------------

#TODO: CONNECT WITH G AND D's DATASET
def training_loop(
    run_dir                 = '.',      # Output directory.
    G_training_set_kwargs   = {},       # Options for G training set.
    D_training_set_kwargs   = {},       # Options for G training set.
    VG_training_set_kwargs  = {},       # Just the dataset of the Cross-Dataset Validation (Low Resolution).
    VD_training_set_kwargs  = {},       # Just the dataset of the Cross-Dataset Validation (High Resolution).
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    lr_scheduler            = None,
    beta2_scheduler         = None,
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    gamma_scheduler         = None,
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 32,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    g_batch_gpu             = 32,        # Number of samples processed at a time by one GPU.
    d_batch_gpu             = 32,        # Number of samples processed at a time by one GPU.
    ema_scheduler           = None,
    aug_scheduler           = None,
    total_kimg              = 1000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 200,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 200,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    metric_eval_ticks       = 200,      # How often to evaluate metrics? None = disable.      
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.(change to True)
    torch.backends.cudnn.allow_tf32 = False              # Improves numerical accuracy.(change to True)
    # torch.backends.cudnn.deterministic = False          # Non deterministic convs for speed
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.
    

    # Load training set.
    if rank == 0:
        print('Loading training set...')

    G_training_set = dnnlib.util.construct_class_by_name(**G_training_set_kwargs) # subclass of training.dataset.Dataset
    G_training_set_sampler = misc.InfiniteSampler(dataset=G_training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    G_training_set_iterator = iter(torch.utils.data.DataLoader(dataset=G_training_set, sampler=G_training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))

    D_training_set = dnnlib.util.construct_class_by_name(**D_training_set_kwargs) # subclass of training.dataset.Dataset
    D_training_set_sampler = misc.InfiniteSampler(dataset=D_training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    D_training_set_iterator = iter(torch.utils.data.DataLoader(dataset=D_training_set, sampler=D_training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))

    if rank == 0:
        print()
        # print('Num images: ', G_training_set_kwargs.max_size + D_training_set_kwargs.max_size)  # i'm not sureee (see R3GAN repo for reference and adjust later)
        print('Num images: ', len(G_training_set) + len(D_training_set)) # try
        print('Image shape:', G_training_set.image_shape)
        print('Label shape:', G_training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=G_training_set.label_dim, img_resolution=G_training_set.resolution)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval() # Generator with Exponential Moving Average of weights

    # Resume from existing pickle.
    if resume_pkl is not None:
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        if rank == 0:
            print(f'Resuming from "{resume_pkl}"')
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
                misc.copy_params_and_buffers(resume_data[name], module, require_all=False)



    # Print network summary tables.
    if rank == 0:
        lr_shape = G_training_set[0][0].shape
        n_lr_img = torch.empty([1, *lr_shape], device=device)
        c = torch.empty([1, G.c_dim], device=device)
        # print('c =', c)
        img = misc.print_module_summary(G, [n_lr_img, c])
        # print('img =',img)
        misc.print_module_summary(D, [img, c])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None

    if (augment_kwargs is not None) and (aug_scheduler is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        
    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    
    opt = dnnlib.util.construct_class_by_name(params=D.parameters(), **D_opt_kwargs)
    if resume_pkl is not None:
        opt.load_state_dict(remap_optimizer_state_dict(resume_data['D_opt_state'], device))
    phases += [dnnlib.EasyDict(name='D', module=D, opt=opt, batch_gpu=d_batch_gpu)]
    
    opt = dnnlib.util.construct_class_by_name(params=G.parameters(), **G_opt_kwargs)
    if resume_pkl is not None:
        opt.load_state_dict(remap_optimizer_state_dict(resume_data['G_opt_state'], device))
    phases += [dnnlib.EasyDict(name='G', module=G, opt=opt, batch_gpu=g_batch_gpu)]
    
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample image.
    grid_size = None
    grid_lr = None
    grid_c = None

    if rank == 0:
        print('Exporting one sample image...')

        # Export only 1 sample
        n_samples = 1
        if len(G_training_set) == 0:
            raise RuntimeError("No samples available in the training set for snapshot export.")

        # Set grid size to (1, 1)
        grid_size = (1, 1)
        print(f"[Debug] Exporting 1 sample, grid size: {grid_size}")

        # Get the 1st sample (LR image only)
        lr_np = G_training_set[0][0]  # (C,H,W) from LR dataset
        hr_np = D_training_set[0][0]  # (C,H,W) from HR datase
        images = np.stack([lr_np])  # [1, C, H, W]
        fname = os.path.splitext(os.path.basename(G_training_set.get_fname(0)))[0] # get the original file name without extension
        save_image_grid(images, os.path.join(run_dir, f"initial_low_resolution.png"), drange=[0, 255], grid_size=grid_size)

        # Dummy label (for conditional generation)
        label = np.zeros(1, dtype=np.int64)
        lr_tensor = torch.from_numpy(lr_np).unsqueeze(0).to(device)  # [1, C, H, W]
        label_tensor = torch.from_numpy(label).to(device)
        grid_lr = [torch.from_numpy(lr_np).to(device)]
        grid_hr = [torch.from_numpy(hr_np).to(device)]
        grid_c  = [torch.zeros(1, dtype=torch.int64, device=device)] 
       
        # Generate fake HR image
        with torch.no_grad():
            fake = G_ema(lr_tensor, label_tensor)
            if isinstance(fake, np.ndarray):
                fake = torch.from_numpy(fake)
            if not isinstance(fake, torch.Tensor):
                raise TypeError(f"G_ema returned type {type(fake)}; expected Tensor or ndarray.")

        # Save fake image
        fake_image = fake.cpu().to(torch.float).numpy()
        save_image_grid(fake_image, os.path.join(run_dir, f"initial_fake.png" ), drange=[-1, 1], grid_size=grid_size) # use same name for both generated and fake (see kine 272)

        print(f"[Debug] Real image shape: {lr_np.shape}")
        print(f"[Debug] Fake image shape: {fake.shape}")

    # i shall return ----------------------------------- (kyr ended this code)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_data['cur_nimg'] if resume_pkl is not None else 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
        
    # Dummy Timing, required to fix phase shift
    for phase in phases:
        if phase.start_event is not None:
            phase.start_event.record(torch.cuda.current_stream(device))
        if phase.end_event is not None:
            phase.end_event.record(torch.cuda.current_stream(device))
        

    while True:
        # Fetch paired LR-HR training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            lr_img, label = next(G_training_set_iterator)
            hr_img, _ = next(D_training_set_iterator)
            # Exceesive memory alloation and tensor operations 
            lr_img = (lr_img.to(device).to(torch.float32) / 127.5 - 1).split(g_batch_gpu)
            hr_img = (hr_img.to(device).to(torch.float32) / 127.5 - 1).split(d_batch_gpu)
            # # Pre normalize data in the dataset 
            # lr_img = lr_img.to(device, non_blocking=True).split(g_batch_gpu)
            # hr_img = hr_img.to(device, non_blocking=True).split(d_batch_gpu)
            
            label = label.to(device).split(g_batch_gpu)

        # Update schedulers.
        cur_lr = cosine_decay_with_warmup(cur_nimg, **lr_scheduler)
        cur_beta2 = cosine_decay_with_warmup(cur_nimg, **beta2_scheduler)
        cur_gamma = cosine_decay_with_warmup(cur_nimg, **gamma_scheduler)
        cur_ema_nimg = cosine_decay_with_warmup(cur_nimg, **ema_scheduler)
        cur_aug_p = cosine_decay_with_warmup(cur_nimg, **aug_scheduler)

        if augment_pipe is not None:
            augment_pipe.p.copy_(misc.constant(cur_aug_p, device=device))

        # Execute training phases.
        for phase in phases:
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            # AccumulateDiscriminatorGradients(gen_z, real_img, real_c, gamma, gain, self.preprocessor) # old
            # for lr, hr, c in zip(lr_img, hr_img, label):

            start_time = time.time()

            for lr, hr, c in zip(lr_img, hr_img, label): # NEW
                loss.accumulate_gradients(
                    phase=phase.name,
                    real_img=hr,
                    real_c=c,
                    gen_z=lr,
                    gamma=cur_gamma,
                    gain=num_gpus * phase.batch_gpu / batch_size
                )

            end_time = time.time()
            
            phase.module.requires_grad_(False)

            for g in phase.opt.param_groups:
                g['lr'] = cur_lr
                g['betas'] = (0, cur_beta2)

            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_beta = 0.5 ** (batch_size / max(cur_ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        cur_nimg += batch_size
        batch_idx += 1

        # --- Add this snippet ---
        if batch_idx % 50 == 0 and rank == 0:
            print(f"iter {batch_idx}, kimg={cur_nimg/1e3:.2f}")

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # iters_per_tick = int(kimg_per_tick * 1000 / batch_size)
        # if rank == 0:
        #     print(f"iters_per_tick: {iters_per_tick}")

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Progress/lr', cur_lr)
        training_stats.report0('Progress/ema_mimg', cur_ema_nimg / 1e6)
        training_stats.report0('Progress/beta2', cur_beta2)
        training_stats.report0('Progress/gamma', cur_gamma)
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')


        # TODO: Later after the evaluation is perfect
        # # Save image snapshot. # OLD
        # if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
        #     images = torch.cat([G_ema(z, c).cpu() for z, c in zip(grid_lr, grid_c)]).to(torch.float).numpy()
        #     save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:09d}.png'), drange=[-1,1], grid_size=grid_size)

        # Save image snapshot. # NEW
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            with torch.no_grad():
                gen_images = []
                real_images = []
                for lr, hr, c in zip(grid_lr, grid_hr, grid_c):  
                    # Generator output (normalized in [-1,1])
                    fake = G_ema(lr.unsqueeze(0).to(device), c.unsqueeze(0).to(device) if c is not None else None)
                    gen_images.append(fake.cpu())
                    
                    # HR dataset image (already in [0,255])
                    real_images.append(hr.unsqueeze(0).cpu())

                # Stack all: [N, C, H, W]
                gen_images = torch.cat(gen_images, dim=0)
                real_images = torch.cat(real_images, dim=0)

            # Save generator outputs ([-1,1])
            save_image_grid(
                gen_images.to(torch.float).numpy(),
                os.path.join(run_dir, f'fakes{cur_nimg//1000:09d}.png'),
                drange=[-1, 1],   # Generator outputs are normalized
                grid_size=grid_size
            )

            # Save real HR images ([0,255])
            save_image_grid(
                real_images.to(torch.float).numpy(),
                os.path.join(run_dir, f'reals{cur_nimg//1000:09d}.png'),
                drange=[0, 255],   # <- FIXED: real dataset is in [0,255]
                grid_size=grid_size
            )


        # Save network snapshot.
        snapshot_in_domain = None
        snapshot_cross_domain = None
        snapshot_pkl = None

        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):

            def prepare_snapshot(G, D, G_ema, G_kwargs, D_kwargs, cur_nimg, phases):
                snapshot = dict(
                    G=G, D=D, G_ema=G_ema,
                    G_training_set_kwargs=copy.deepcopy(G_kwargs),
                    D_training_set_kwargs=copy.deepcopy(D_kwargs),
                    cur_nimg=cur_nimg,
                )
                for phase in phases:
                    snapshot[phase.name + '_opt_state'] = remap_optimizer_state_dict(phase.opt.state_dict(), 'cpu')
                for key, value in list(snapshot.items()):
                    if isinstance(value, torch.nn.Module):
                        value = copy.deepcopy(value).eval().requires_grad_(False)
                        if num_gpus > 1:
                            misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                            for param in misc.params_and_buffers(value):
                                torch.distributed.broadcast(param, src=0)
                        snapshot[key] = value.cpu()
                return snapshot


            # (A) In-domain snapshot
            snapshot_in_domain = prepare_snapshot(G, D, G_ema, G_training_set_kwargs, D_training_set_kwargs, cur_nimg, phases)

            # (B) Cross-domain snapshot
            snapshot_cross_domain = prepare_snapshot(G, D, G_ema, VG_training_set_kwargs, VD_training_set_kwargs, cur_nimg, phases)

            # Save ONE snapshot file (let’s keep cross-domain)
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:09d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_cross_domain, f)

        # Evaluate metrics.
        if (metric_eval_ticks is not None) and (done or cur_tick % metric_eval_ticks == 0):
            if (snapshot_in_domain is not None) and (snapshot_cross_domain is not None) and (len(metrics) > 0):
                if rank == 0:
                    print('Evaluating metrics...')

                # (A) In-domain Validation
                for metric in metrics:
                    result_dict = metric_main.calc_metric(
                    metric=metric, 
                    G=snapshot_in_domain['G_ema'],
                    G_dataset_kwargs =  G_training_set_kwargs,
                    D_dataset_kwargs =  D_training_set_kwargs, 
                    num_gpus=num_gpus, rank=rank, device=device
                )
                    if rank == 0:
                        metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                    stats_metrics.update({f"{metric}_in_domain": result_dict.results})

                # (B) Cross-domain Validation
                # for metric in metrics:
                #     result_dict = metric_main.calc_metric(
                #         metric=metric, 
                #         G=snapshot_cross_domain['G_ema'],
                #         G_dataset_kwargs =  VG_training_set_kwargs,
                #         D_dataset_kwargs =  VD_training_set_kwargs, 
                #         num_gpus=num_gpus, rank=rank, device=device
                #     )

                #     if rank == 0:
                #         metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                #     stats_metrics.update({f"{metric}_cross_domain": result_dict.results})

        # Cleanup
        del snapshot_in_domain, snapshot_cross_domain


        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            # for name, value in stats_metrics.items(): # OLD
            #     stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items(): # NEW
                if isinstance(value, dict):
                    for k, v in value.items():
                        stats_tfevents.add_scalar(f'Metrics/{name}/{k}', v, global_step=global_step, walltime=walltime)
                else:
                    stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)

            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
