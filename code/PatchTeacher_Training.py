"""
This script simply points RFDETR at the patch dataset and trains it using a standard training loop.

The hyperparameters are optimized for performance on my local machine, tuning may be required for other environments.
    GPU: 4090 w / 24Gb VRAM
    RAM: 128Gb

Main choke point for training is currently the GPU.
"""


import os
from rfdetr import RFDETRBase  # or RFDETRSmall / Medium etc.

from pytorch_lightning.loggers import TensorBoardLogger

from datetime import datetime


if __name__ == "__main__":
    
    data_dir = os.path.abspath("Patch_Data")
    dataset_dir = os.path.join(data_dir, "dataset")  # directory containing train/, valid/, test/
    train_dir = os.path.join(dataset_dir, "train")
    valid_dir = os.path.join(dataset_dir, "valid")
    test_dir = os.path.join(dataset_dir, "test")
    
    os.makedirs(os.path.join(dataset_dir, 'output'), exist_ok = True)
    output_dir = os.path.join(dataset_dir, 'output')
    

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")+'_dropPath'

    logger = TensorBoardLogger(
        save_dir="logs",
        name="HelldiverII",
        version=run_id
    )


    
    
    sizes = ["small", "medium", "large", "massive"]
    types = ["bot", "bug", "squid"]
    class_names = [f"{s}_{t}" for s in sizes for t in types]

    model = RFDETRBase(num_classes = len(class_names), resolution=896)
    
    
    
    lr = 1e-4
# =============================================================================
#     schedule = {
#         "type": "cosine_restart",
#         "warmup_epochs": 5,
#         "period": 50,      # long first cycle, slows convergence
#         "min_lr": 1e-6,
#         "t_mult": 2        # doubles cycle length each restart
#         }
# =============================================================================
    schedule = {"type": "cosine", "min_lr": 1e-6}
    # schedule = {"type": "cosine", "warmup_epochs": 5, "min_lr": 1e-6}
    
    model.train(
        dataset_dir=dataset_dir,
        epochs=200,
    
        # Slightly smaller effective batch = less memorization
        batch_size=8,
        grad_accum_steps=4,   # total batch = 32 instead of 40
    
        lr=lr,
        lr_encoder=lr * 0.6,  # slower encoder adaptation
        lr_schedule=schedule,
    
        output_dir=output_dir,
        device='cuda',
        num_workers=8,
        class_names=class_names,
    
        eval_interval=5,
        save_best=True,
    
        # Stability
        use_amp=True,
        max_grad_norm=0.3,    # tighter clipping
        ema=True,
        
        num_select=25,
        
        multi_scale=True,
        expanded_scales=False,
        
        drop_path=0.15,
        weight_decay=2e-4,
    
        # Aspect ratio safe
        do_random_resize_via_padding=True,
    
        checkpoint_interval=5,
        
        early_stopping=True,
        early_stopping_patience=12,
        early_stopping_min_delta=0.002,
    
        logger=logger,
        # resume=os.path.join(output_dir, "checkpoint.pth")
    )