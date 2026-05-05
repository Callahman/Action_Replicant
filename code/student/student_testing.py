import bbox_student
import numpy as np
import os
import torch

# Import paths (for directory management)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # up 2 levels → Main_Dir
import paths


teacher_dir = paths.TEACHER_DATA_DIR
img_dir = paths.IMAGE_DIR

teacher_files = os.listdir(teacher_dir)
np.random.shuffle(teacher_files)
# teacher_files = teacher_files[:10] # for quick testing

# teacher_files = ['2025_11_13_02_39_21_186996_teacher.json']


transform = bbox_student.get_transforms(train=True)
ds = bbox_student.BBoxDataset(teacher_files, teacher_dir, img_dir, transform=transform)
dl = torch.utils.data.DataLoader(ds, batch_size=3, shuffle=False)


model = bbox_student.CustomYolov8n(num_classes=12)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

CIOU = bbox_student.CustomYoloLoss().ciou_loss
FocalLoss = bbox_student.CustomYoloLoss().focal_loss
tal = bbox_student.CustomYoloLoss().TAL

count = 0
for batch in dl:
    optimizer.zero_grad()
    img, x = batch

    bbox = x['boxes']
    labels = x['labels']
    confs = x['confs']

    img = img.cuda() if torch.cuda.is_available() else img
    yhat = model(img)

    head, (p3, p4, p5) = yhat

    yhat_boxes = head[:, :4, :]
    yhat_classes = head[:, 4:, :]

    target_boxes = bbox.cuda() if torch.cuda.is_available() else bbox
    target_boxes = target_boxes.permute(0, 2, 1)  # [B, 4, Y]
    target_classes = labels.cuda() if torch.cuda.is_available() else labels
    target_confs = confs.cuda() if torch.cuda.is_available() else confs

    # Match predicted boxes to ground truth boxes using TAL and refine to just matches
    matched, gt_idx = tal(yhat_boxes, yhat_classes, target_boxes, target_classes)
    # yhat_boxes, target_boxes, yhat_classes, target_classes, target_confs = bbox_student.CustomYoloLoss().gt_assignment(matched, gt_idx, yhat_boxes, target_boxes, yhat_classes, target_classes, target_confs)
    yhat_boxes, target_boxes, yhat_classes, target_classes, target_confs = bbox_student.CustomYoloLoss().gt_assignment(matched, gt_idx, yhat_boxes, target_boxes, yhat_classes, target_classes)
        
    print(yhat_boxes.shape, target_boxes.shape, yhat_classes.shape, target_classes.shape)

    # Test CIOU loss
    ciou_loss = CIOU(yhat_boxes, target_boxes)
    
    # Test FocalLoss
    focal_loss = FocalLoss(yhat_classes, target_classes, gamma=2.0)

    loss = ciou_loss * 7.5 + focal_loss * 0.5
    print(loss.item(), '\t', ciou_loss.item(), '\t', focal_loss.item())
    loss.backward()
    optimizer.step()

    count+=1
    if count >= 100:
        break