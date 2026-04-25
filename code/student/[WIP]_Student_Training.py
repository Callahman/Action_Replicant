"""
Work in Progress: Student Training Loop for Custom YOLOv8n

Goal: Create a custom training loop for a YOLOv8n for future inference
Will also experiement with custom loss functions and custom transformations to improve model performance
"""


import random
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# This refers to the custom library "[WIP]_bbox_student.py"
from bbox_student import BBoxDataset, get_transforms

from ultralytics import YOLO
from bbox_student import CustomYoloLoss, CustomYolov8n

# Import paths (for directory management)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # up 2 levels → Main_Dir
import paths

# ============================================================
# Config
# ============================================================

sizes = ["small", "medium", "large", "massive"]
types = ["bot", "bug", "squid"]
class_names = [f"{s}_{t}" for s in sizes for t in types]



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_CLASSES = len(class_names)
MAX_OBJECTS = 40
NUM_ANCHORS = 3 # Should not be changed. Default for YOLO
EPOCHS = 5
LR = 1e-3



img_dir = paths.IMAGE_DIR
teacher_dir = paths.TEACHER_DIR
teacher_files = os.listdir(teacher_dir)

# Parameters
train_ratio = 0.8  # for example, 80% training
random_seed = 42   # fixed seed for reproducibility

random.seed(random_seed)
random.shuffle(teacher_files)

num_train = int(len(teacher_files) * train_ratio)

train_files = teacher_files[:num_train]
valid_files = teacher_files[num_train:]



# ============================================================
# Dataset / Dataloader
# ============================================================
train_dataset = BBoxDataset(train_files, teacher_dir, img_dir, transform=get_transforms(train=True), max_objects=MAX_OBJECTS)
valid_dataset = BBoxDataset(valid_files, teacher_dir, img_dir, transform=get_transforms(train=False), max_objects=MAX_OBJECTS)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# Model
# ============================================================
# model = YOLO("yolov8n.pt").model
model = CustomYolov8n()
model = model.to(DEVICE)


yolo_loss = CustomYoloLoss()


optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ============================================================
# Training Loop
# ============================================================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    
    
    for images, targets in train_loader:
            
        
        images = images.to(DEVICE)
        confs = targets["confs"].to(DEVICE)
        labels = targets["labels"].to(DEVICE)
        bboxes = targets["boxes"].to(DEVICE)

        optimizer.zero_grad()

        pred = model(images)
        
        p3_pred = pred[0]
        p4_pred = pred[1]
        p5_pred = pred[2]
        
        ### Assign target_boxes
        p3_box, p3_labels, p3_mask, p4_box, p4_labels, p4_mask, p5_box, p5_labels, p5_mask = yolo_loss.assignment(bboxes, labels)
        
        pred_p3_box, pred_p3_label = yolo_loss.decode(p3_pred)
        pred_p4_box, pred_p4_label = yolo_loss.decode(p4_pred)
        pred_p5_box, pred_p5_label = yolo_loss.decode(p5_pred)
        
        loss = yolo_loss.pairwise_ciou(pred_p3_box, p3_box, eps=1e-7)
        
        print(loss.shape)
        
# =============================================================================
#         cls_loss = classification_loss_fn(class_logits, labels)
#         box_loss = bbox_loss_fn(bbox_preds, bboxes)
# 
#         loss = cls_loss + box_loss
#         loss.backward()
#         optimizer.step()
# 
#         total_loss += loss.item()
# =============================================================================
        
        break
    break

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}")

print("Training complete.")
