"""
This process takes the original image dataset and pre-processes all the images for the teacher training
The pre-processing speeds up training time and reduces the work needed to create a custom transform

For each image:
    - Establish a fixed grid of slightly overlapping pathches, save these as new images
    - For each object in the image, create a patch around the object
        - This is done so empty images don't dominate the training dataset
        - A jitter is applied around the object to prevent overfitting to perfectly centered objects
    - For each new patch, create a COCO-style (native for RFDETR) annotation for the bounding boxes
        - Box locations are updated based on their position within the new patched image
"""


# %% Import libraries
import os
from PIL import Image
import pickle
from tqdm import tqdm
import random
import json
import numpy as np

# Import paths (for directory management)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # up 2 levels → Main_Dir
import paths







# %% Define globals
dataset_dir = paths.DATASET_DIR

patch_dir = paths.PATCH_DIR
patch_dataset_dir = paths.PATCH_DATASET_DIR




# Configuration parameters
# --- Tile configuration ---
PATCH_GRID_ROWS = 3
PATCH_GRID_COLS = 5
PATCH_GRID_OVERLAP = 0.15  # Fractional overlap for normal grid tiles
PATCH_WIDTH = 896          # Fixed width for all tiles
PATCH_HEIGHT = 896         # Fixed height for all tiles

# --- Jitter configuration ---
JITTER_FACTOR = 0.2        # Fraction of object size for jitter


# =============================================================================
# patch_info = {
#     0: (0.0, 0.0, 1024, 1024),
#     1: (870.4, 0.0, 1024, 1024),
#     2: (1740.8, 0.0, 1024, 1024),
#     3: (2611.2, 0.0, 1024, 1024),
#     4: (2818, 0.0, 1024, 1024),
#     5: (0.0, 870.4, 1024, 1024),
#     6: (870.4, 870.4, 1024, 1024),
#     7: (1740.8, 870.4, 1024, 1024),
#     8: (2611.2, 870.4, 1024, 1024),
#     9: (2818, 870.4, 1024, 1024),
#     10: (0.0, 1138, 1024, 1024),
#     11: (870.4, 1138, 1024, 1024),
#     12: (1740.8, 1138, 1024, 1024),
#     13: (2611.2, 1138, 1024, 1024),
#     14: (2818, 1138, 1024, 1024)
#     }
# =============================================================================


# %% 

count = 0
for current_dir in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(patch_dataset_dir, current_dir), exist_ok=True)

    annotate = json.load(open(os.path.join(dataset_dir, current_dir, '_annotations.coco.json'), 'r'))

    # Create new annotation structure if missing
    if '_annotations.coco.json' not in os.listdir(os.path.join(patch_dataset_dir, current_dir)):
        new_annotate = {
            "info": {"description": "Manual bounding box classification"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": annotate['categories']
        }
    else:
        new_annotate = json.load(open(os.path.join(patch_dataset_dir, current_dir, '_annotations.coco.json'), 'r'))

    # Track processed images and annotation IDs
    completed = []
    img_id = 1
    annotate_id = max([0] + [a['id'] for a in new_annotate['annotations']]) + 1
    for i in new_annotate['images']:
        completed.append(i['file_name'].split('_patch_')[0] + '_screen.jpg')
        img_id = max(i['id'], img_id) + 1
    completed = list(set(completed))

    # Build image->annotation lookup
    img_annotations = {}
    for i, a in enumerate(annotate['annotations']):
        img_annotations.setdefault(a['image_id'], []).append(i)

    # Iterate over all images
    for i in tqdm(annotate['images']):
        file = i['file_name']
        if file in completed:
            continue

        img = Image.open(os.path.join(dataset_dir, current_dir, file))
        img_width, img_height = img.size
        
        # --- Fixed-size tiling with overlap across the whole image ---
        patch_info = {}
        patch_id = 0
        
        step_x = int(PATCH_WIDTH * (1 - PATCH_GRID_OVERLAP))
        step_y = int(PATCH_HEIGHT * (1 - PATCH_GRID_OVERLAP))
        
        # Number of patches needed to fully cover width/height
        num_patches_x = (img_width - 1) // step_x + 1
        num_patches_y = (img_height - 1) // step_y + 1
        
        for r in range(num_patches_y):
            for c in range(num_patches_x):
                x1 = c * step_x
                y1 = r * step_y
        
                # Ensure the patch stays inside the image
                x1 = min(x1, img_width - PATCH_WIDTH)
                y1 = min(y1, img_height - PATCH_HEIGHT)
        
                patch_info[patch_id] = [x1, y1, PATCH_WIDTH, PATCH_HEIGHT]
                patch_id += 1
                patch_id += 1
                
                
                
        # --- 2) Add jittered tiles per object, scaled by object count ---
        num_objects = len(img_annotations.get(i['id'], []))
        if num_objects > 0:
            if num_objects == 1:
                n_jitter = 4
            elif num_objects <= 3:
                n_jitter = 3
            elif num_objects <= 5:
                n_jitter = 2
            else:
                n_jitter = 1
        else:
            n_jitter = 0  # No objects → no jittered tiles
        
        for ai in img_annotations.get(i['id'], []):
            a = annotate['annotations'][ai]
            box = a['bbox']
            bx1, by1, bw, bh = box
            bx2 = bx1 + bw
            by2 = by1 + bh
                
            for nj in range(n_jitter):
                # Compute jitter based on object size + patch size
                jitter_x = np.clip(np.random.normal(0, 0.18), -0.5, 0.5) * PATCH_WIDTH
                jitter_y = np.clip(np.random.normal(0, 0.18), -0.5, 0.5) * PATCH_HEIGHT
        
                # Center object in patch and apply jitter
                x1_j = int((bx1 + bx2)/2 - PATCH_WIDTH//2 + jitter_x)
                y1_j = int((by1 + by2)/2 - PATCH_HEIGHT//2 + jitter_y)
        
                # Clip to image boundaries
                x1_j = max(0, min(x1_j, img_width - PATCH_WIDTH))
                y1_j = max(0, min(y1_j, img_height - PATCH_HEIGHT))
        
                # Save patch info
                patch_info[patch_id] = [x1_j, y1_j, PATCH_WIDTH, PATCH_HEIGHT]
                patch_id += 1


        # --- 3) Crop and save patches, create new COCO entries ---
        for pi in patch_info:
            p = patch_info[pi]
            x1, y1, w, h = [int(round(v)) for v in p]
            x2, y2 = x1 + w, y1 + h
            patch = img.crop((x1, y1, x2, y2))

            patch_img_file = file.replace('screen.jpg', f'_patch_{pi}_screen.jpg')
            patch.save(os.path.join(patch_dataset_dir, current_dir, patch_img_file))

            img_data = {
                'id': img_id,
                'file_name': patch_img_file,
                'height': h,
                'width': w
            }
            new_annotate['images'].append(img_data)

            # Add annotations
            for ai in img_annotations.get(i['id'], []):
                a = annotate['annotations'][ai]
                box = a['bbox']
                cat = a['category_id']
                crowd = a['iscrowd']

                bx1, by1, bw, bh = box
                bx2 = bx1 + bw
                by2 = by1 + bh

                # Map to patch coordinates
                bx1_patch = max(0, bx1 - x1)
                by1_patch = max(0, by1 - y1)
                bx2_patch = min(w, bx2 - x1)
                by2_patch = min(h, by2 - y1)

                b_width = bx2_patch - bx1_patch
                b_height = by2_patch - by1_patch

                if b_width <= 0 or b_height <= 0:
                    continue

                anno_data = {
                    'id': annotate_id,
                    'image_id': img_id,
                    'category_id': cat,
                    'bbox': [bx1_patch, by1_patch, b_width, b_height],
                    'area': b_width * b_height,
                    'iscrowd': crowd
                }
                new_annotate['annotations'].append(anno_data)
                annotate_id += 1

            img_id += 1

        count += 1
        if count % 100 == 0:
            json.dump(new_annotate, open(os.path.join(patch_dataset_dir, current_dir, '_annotations.coco.json'), 'w'))

    # Final write
    json.dump(new_annotate, open(os.path.join(patch_dataset_dir, current_dir, '_annotations.coco.json'), 'w'))
