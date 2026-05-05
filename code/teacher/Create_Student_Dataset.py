"""
This process takes the trained teacher from "PatchTeacher_Training.py" and labels each image for the student

Steps:
    1. Import libraries, instantiate globals, yada yada
    2. Check for any existing teacher labels. Consider these "true" labels and use them instead of inference
    3. For each image without existing labels, run batched patch inference to generate teacher labels
        - Patched inferences are adjusted to account for patch position
        - Patched inferences are coalated into one set of predictions per image
        - Duplicate predictions are not removed, YOLO should converge on the correct output during training
    4. Save the labels to the Teacher directory

Note: This process can be extremely time consuming, as it runs inference across the whole dataset.
    Even with batching and GPU acceleration it can take tens of hours depending on your dataset size

Future optimizations to consider:
    - Implement multiprocessing to load images
    - Implement batch-wise patching and remove for-loops. Replace with vectorization (striding window) to generate patches

Batch size capped at 6 images per batch
GPU (4090) memeory usage caps at 6 images w/ 896x896 patch size & 20% overlap between patches
Roughly nets to 54 patches per batch
"""


import os
import cv2
import torch
import numpy as np
from rfdetr import RFDETRBase
from itertools import product
import json
from tqdm import tqdm

# Import paths (for directory management)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # up 2 levels → Main_Dir
import paths


torch.set_grad_enabled(False)

BATCH_SIZE = 6  # number of images to batch together



# ---------------- TILING ----------------
def tile_image(image, patch_size, overlap=0.2):
    h, w, _ = image.shape
    stride = max(int(patch_size * (1 - overlap)), 1)  # ensure stride >= 1

    xs = range(0, w, stride)
    ys = range(0, h, stride)

    patches = []
    for y, x in product(ys, xs):
        patch = image[y:y + patch_size, x:x + patch_size]

        # Pad edges if patch is smaller
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            patch = np.pad(
                patch,
                ((0, patch_size - patch.shape[0]),
                 (0, patch_size - patch.shape[1]),
                 (0, 0)),
                mode="constant"
            )

        patches.append((patch, x, y))
        
    return patches


# ---------------- INFERENCE ----------------
@torch.no_grad()
def run_batch_inference(image_paths):
    """Run batch inference on a list of image paths."""
    image_shapes = []
    patches = []
    offsets = []
    img_names = []
    # Step 1: Read images and tile them
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"OpenCV could not read image: {image_path}")

        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tiles = tile_image(image, PATCH_SIZE)
        
        for i, t in enumerate(tiles):
            patch = t[0]
            offset = (t[1], t[2])
            
            patches.append(patch)
            offsets.append(offset)
            image_shapes.append(image.shape[:2])
            img_names.append(image_path)

    # Step 2: Stack all tiles and run batch inference
    batch_preds = model.predict(patches, conf_threshold=0.3, device=DEVICE)
    
    # Step 3: Split predictions back to images
    # Initialize empty data for each image
    results = {}
    for preds, shape, (offset_x, offset_y), img_name in zip(batch_preds, image_shapes, offsets, img_names):
        
        if img_name not in results:
            results[img_name] = {
                'coords': [],
                'centered': [],
                'class': [],
                'conf': [],
                'true': 0,
                'img_height': shape[0],
                'img_width': shape[1]
            }

        for det in preds:
            x1, y1, x2, y2 = det[0]
            true_x1 = int(x1 + offset_x)
            true_x2 = int(x2 + offset_x)
            true_y1 = int(y1 + offset_y)
            true_y2 = int(y2 + offset_y)
            true_w = int(true_x2 - true_x1)
            true_h = int(true_y2 - true_y1)
            true_cx = int(true_x1 + true_w / 2)
            true_cy = int(true_y1 + true_h / 2)
            class_id = int(det[3])-1
            conf = float(det[2])

            results[img_name]['coords'].append([true_x1, true_y1, true_x2, true_y2])
            results[img_name]['centered'].append([true_cx, true_cy, true_w, true_h])
            results[img_name]['class'].append(class_id)
            results[img_name]['conf'].append(conf)

    return results


# ---------------- CONFIG ----------------
PATCH_SIZE = 896
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = os.path.join(paths.PATCH_OUTPUT_DIR, "checkpoint.pth")

# IMPORTANT: raw string for Windows paths
image_dir = paths.IMAGE_DIR
teacher_dir = paths.TEACHER_DATA_DIR
dataset_dir = paths.DATASET_DIR


class_names = {
    0:'small_squid',
    1:"small_bot",
    2:'small_bug',
    3:'medium_squid',
    4:'medium_bot',
    5:'medium_bug',
    6:'large_squid',
    7:'large_bot',
    8:'large_bug',
    9:'massive_squid',
    10:'massive_bot',
    11:'massive_squid'
    }


# ---------------- LOAD MODEL ----------------
model = RFDETRBase(
    num_classes=len(list(class_names.keys())),
    resolution=PATCH_SIZE,
    pretrain_weights=checkpoint_path,
    device=DEVICE
)
model.model.model.eval()
# model.optimize_for_inference()

model.model.model = torch.compile(
    model.model.model,
    backend="aot_eager"
)


# ---------------- LOAD TRUE LABELS ----------------
true_labels = {}
for d in ['train','test','valid']:
    print(f'\nLoading true labels from: {d}')
    yml = json.load(open(os.path.join(dataset_dir, d, '_annotations.yolo.json'),'r'))
    
    temp = {}
    for a in tqdm(yml['annotations']):
        if a['image_id'] in list(temp.keys()):
            temp[a['image_id']].append(a)
        else:
            temp[a['image_id']] = [a]
            
    for i in tqdm(yml['images']):
        if i['id'] in list(temp.keys()):
            true_labels[i['file_name']] = {
                'annotations':temp[i['id']],
                'img':{
                    'height':i['height'],
                    'width':i['width']
                    }
                }
            
        else:
            true_labels[i['file_name']] = {
                'annotations':[],
                'img':{
                    'height':i['height'],
                    'width':i['width']
                    }
                }
    



# Refine list of files to process - only those that don't have labels yet
print('Prepping teacher label list')
og_teacher_files = set(os.listdir(teacher_dir))
og_img_files = os.listdir(image_dir)

teacher_files = []
img_files = []
for fname in tqdm(og_img_files):
    teacher_file = fname.replace('screen.jpg','teacher.json')
    if teacher_file not in og_teacher_files:
        teacher_files.append(teacher_file)
        img_files.append(fname)


# Run trough dataset and generate teacher labels (patch image, run inference, coalate result, save to teacher data dir)
print('Running batch inference')
batch_paths = []
batch_teacher_files = []
for fname, teacher_file in tqdm(zip(img_files, teacher_files), total=len(img_files)):
    
    ### Check for true labels, use those if available
    if fname in list(true_labels.keys()):
        
        data = {
            'coords':[],
            'centered':[],
            'class':[],
            'conf':[],
            'true':1,
            'img_height':None,
            'img_width':None
            }
        
        labels = true_labels[fname]
        data['img_height'] = labels['img']['height']
        data['img_width'] = labels['img']['width']
        
        
        for a in labels['annotations']:
            x1, y1, w, h = a['bbox']
            x2 = x1 + w
            y2 = y1 + h
            xc = int(round(x1 + (w / 2), 0))
            yc = int(round(y1 + (h / 2), 0))
            
            data['coords'].append([x1, y1, x2, y2])
            data['centered'].append([xc, yc, w, h])
            data['class'].append(a['category_id']-1)
            data['conf'].append(1)
            
        json.dump(data, open(os.path.join(teacher_dir, teacher_file), 'w'))
            
    ### If no true labels, then generate teacher labels
    else:
        batch_paths.append(os.path.join(image_dir, fname))
        batch_teacher_files.append(teacher_file)
    
        if len(batch_paths) == BATCH_SIZE:
            batch_results = run_batch_inference(batch_paths)
            
            for img_file, tfile in zip(batch_paths, batch_teacher_files):
                result = batch_results[img_file]
                json.dump(result, open(os.path.join(teacher_dir, tfile), 'w'))

                
            batch_paths = []
            batch_teacher_files = []

# Run remaining images
if batch_paths:
    batch_results = run_batch_inference(batch_paths)
    
    for img_file, tfile in zip(batch_paths, batch_teacher_files):
        result = batch_results[img_file]
        json.dump(result, open(os.path.join(teacher_dir, tfile), 'w'))


    