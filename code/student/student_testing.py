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

transform = bbox_student.get_transforms(train=True)
ds = bbox_student.BBoxDataset(teacher_files, teacher_dir, img_dir, transform=transform)


model = bbox_student.CustomYolov8n(num_classes=12)
model.eval()

CIOU = bbox_student.CustomYoloLoss().ciou_loss
tal = bbox_student.CustomYoloLoss().TAL

count = 0
for batch in ds:
    img, x = batch

    bbox = x['boxes'].unsqueeze(0)
    labels = x['labels'].unsqueeze(0)
    confs = x['confs'].unsqueeze(0)


    ####################################################################### TEMP
    if torch.sum(labels >= 0) == 0:
        continue
    ####################################################################### TEMP

    with torch.no_grad():
        img = img.cuda() if torch.cuda.is_available() else img
        yhat = model(img.unsqueeze(0))

    head, (p3, p4, p5) = yhat

    yhat_boxes = head[:, :4, :]
    yhat_classes = head[:, 4:, :]

    target_boxes = bbox.cuda() if torch.cuda.is_available() else bbox
    target_boxes = target_boxes.permute(0, 2, 1)  # [B, 4, Y]
    target_classes = labels.cuda() if torch.cuda.is_available() else labels

    matched, gt_idx = tal(yhat_boxes, yhat_classes, target_boxes, target_classes)



    #######################################################################
    ### Evaluate that this is correctly matching pred to gt
    ### Concern: The reshape may cause issues later when batch_size > 1
    ### Test with dataloader
    print(torch.sum(matched))
    print(yhat_boxes.shape, target_boxes.shape)
    matched_gt = gt_idx[matched]
    print(matched_gt.shape, matched.shape)
    training_boxes = target_boxes[:, :, matched_gt].reshape(-1, 4)
    yhat_boxes = yhat_boxes.permute(0, 2, 1)[matched]
    print(yhat_boxes.shape, target_boxes.shape)
    
    # Test CIOU loss
    if torch.sum(labels >= 0) == 0:
        ciou_loss = torch.tensor(0.0)
    else:
        print(yhat_boxes.shape, training_boxes.shape)
        ciou_loss = CIOU(yhat_boxes, training_boxes)
    
    # Test BCELoss + BCE penalty on NON-Matched boxes
    # Will also need to implement some measure of teacher confidence to weight the loss contribution of each box

    print(ciou_loss)

    break