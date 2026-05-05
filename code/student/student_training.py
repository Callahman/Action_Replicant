"""
Student Training Loop for Retraining YOLOv8n on custom data

Goal: Create a custom training loop for a YOLOv8n for future inference
Creates pipeline to retrain a YOLOv8n
"""

import bbox_student
import random
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

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
BATCH_SIZE = 256
NUM_WORKERS = 4
PREFETCH_FACTOR = 4
NUM_CLASSES = len(class_names)
# MAX_OBJECTS = 40
NUM_ANCHORS = 3 # Should not be changed. Default for YOLO
EPOCHS = 50
LR = 1e-3
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 3



img_dir = paths.IMAGE_DIR
teacher_dir = paths.TEACHER_DATA_DIR
teacher_files = os.listdir(teacher_dir)

student_output_dir = paths.STUDENT_DATA_DIR
student_models_dir = paths.STUDENT_MODELS_DIR

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
train_dataset = bbox_student.BBoxDataset(train_files, teacher_dir, img_dir, transform=bbox_student.get_transforms(train=True))
valid_dataset = bbox_student.BBoxDataset(valid_files, teacher_dir, img_dir, transform=bbox_student.get_transforms(train=False))


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR
)


# ============================================================
# Model
# ============================================================
model = bbox_student.CustomYolov8n(num_classes=12, device = DEVICE)
model.train()


optimizer = torch.optim.Adam(model.parameters(), lr=LR)

custom_yolo = bbox_student.CustomYoloLoss()


# ============================================================
# Training Loop
# ============================================================
if __name__ == '__main__':
    track_metrics = {}
    valid_loss_count = 0
    for epoch in range(EPOCHS):
        print(f'Epoch: {epoch+1}')


        ### Begin training loop
        train_metrics = {
            "loss": 0,
            "ciou_loss": 0,
            "focal_loss": 0,
            "total_counts": 0
            }
        model.train()
        for images, data in tqdm(train_loader):
            
            # Render data to device
            images = images.to(DEVICE)
            confs = data["confs"].to(DEVICE)
            labels = data["labels"].to(DEVICE)
            bboxes = data["boxes"].to(DEVICE)

            optimizer.zero_grad()

            head, (p3, p4, p5) = model(images)

            # Structure the data for loss computation
            bboxes = bboxes.permute(0, 2, 1)
            yhat_boxes = head[:, :4, :]
            yhat_labels = head[:, 4:, :]

            # Match predicted boxes to ground truth boxes using TAL and refine to just matches
            matched, gt_idx = custom_yolo.TAL(yhat_boxes, yhat_labels, bboxes, labels)

            yhat_boxes, target_boxes, yhat_classes, target_classes, target_confs = (
                custom_yolo.gt_assignment(matched, gt_idx, yhat_boxes, bboxes, yhat_labels, labels)
                )
            
            # Calculate weighted losses
            loss, ciou_loss, focal_loss = custom_yolo.loss(yhat_boxes, target_boxes, yhat_labels, target_classes)
            
            train_metrics['loss'] += loss.item()
            train_metrics['ciou_loss'] += ciou_loss.item()
            train_metrics['focal_loss'] += focal_loss.item()
            train_metrics['total_counts'] += images.shape[0]
            
            loss.backward()
            optimizer.step()


        ### Begin validation loop
        model.eval()
        with torch.no_grad():
            valid_metrics = {
                "loss": 0,
                "ciou_loss": 0,
                "focal_loss": 0,
                "total_counts": 0
            }
            for images, data in tqdm(valid_loader):
                # Render data to device
                images = images.to(DEVICE)
                confs = data["confs"].to(DEVICE)
                labels = data["labels"].to(DEVICE)
                bboxes = data["boxes"].to(DEVICE)

                head, (p3, p4, p5) = model(images)

                # Structure the data for loss computation
                bboxes = bboxes.permute(0, 2, 1)
                yhat_boxes = head[:, :4, :]
                yhat_labels = head[:, 4:, :]

                # Match predicted boxes to ground truth boxes using TAL and refine to just matches
                matched, gt_idx = custom_yolo.TAL(yhat_boxes, yhat_labels, bboxes, labels)

                yhat_boxes, target_boxes, yhat_classes, target_classes, target_confs = (
                    custom_yolo.gt_assignment(matched, gt_idx, yhat_boxes, bboxes, yhat_labels, labels)
                    )
                
                # Calculate weighted losses
                loss, ciou_loss, focal_loss = custom_yolo.loss(yhat_boxes, target_boxes, yhat_labels, target_classes)

                valid_metrics['loss'] += loss.item()
                valid_metrics['ciou_loss'] += ciou_loss.item()
                valid_metrics['focal_loss'] += focal_loss.item()
                valid_metrics['total_counts'] += images.shape[0]


        track_metrics[epoch] = {'train': train_metrics, 'valid': valid_metrics}
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {(train_metrics['loss'] / train_metrics['total_counts']):.4f} Valid Loss: {(valid_metrics['loss'] / valid_metrics['total_counts']):.4f}")
        print(f"CIOU: {(train_metrics['ciou_loss'] / train_metrics['total_counts']):.4f} Valid CIOU: {(valid_metrics['ciou_loss'] / valid_metrics['total_counts']):.4f}")
        print(f"Focal: {(train_metrics['focal_loss'] / train_metrics['total_counts']):.4f} Valid Focal: {(valid_metrics['focal_loss'] / valid_metrics['total_counts']):.4f}")
        print('\n')


        ### Evaluate Train vs Valid Metrics and Save Model
        train_loss = train_metrics['loss'] / train_metrics['total_counts']
        valid_loss = valid_metrics['loss'] / valid_metrics['total_counts']
        if valid_loss > train_loss:
            valid_loss_count += 1
        else:
            valid_loss_count = 0

        if valid_loss_count >= EARLY_STOPPING_PATIENCE and EARLY_STOPPING:
            print(f"Validation loss has exceeded training loss for {valid_loss_count} consecutive epochs. Stopping training to prevent overfitting.")
            break
        else:
            print(f"Validation loss is within acceptable range. Saving model for epoch {epoch+1}...")
            torch.save({'model_state_dict':model.state_dict(), 'optimizer_state_dict':optimizer.state_dict(), 'metrics':track_metrics}, student_models_dir / f"yolov8n_epoch{epoch+1}.pth")
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), student_models_dir / "yolov8n_checkpoint.pth")
        
    print("Training complete.")
