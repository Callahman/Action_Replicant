
"""
Custom library to be used for student training.
Since this student will be used downstream as an input for the agent, I want more control over it.
    This library is used to take the packaged YOLO model, and create a custom pipeline:
        - Pytorch Model
        - Dataset for training/inference
        - Loss function(s)
        - Etc


Sorry about the mess!
This is a work in progress!
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import math
from torchvision.transforms import v2 as T
from torchvision.tv_tensors import BoundingBoxes
from ultralytics import YOLO


# ============================================================
# Transforms
# ============================================================
def get_transforms(train=True):
    if train:
        transforms = T.Compose([
            # Resize first (bounding boxes are scaled)
            T.Resize((640, 360)),

            # Pad to square using torchvision.transforms.v2 Pad
            T.Pad(
                padding=(0, 280),
                padding_mode="constant",  # automatically pads shorter side to match longer
                fill=0              # optional: background color
            ),

            # Random horizontal flip (affects bboxes)
            T.RandomHorizontalFlip(p=0.5),

            # Random affine transform (translation, scale, shear)
            T.RandomAffine(
                degrees=0,                # no rotation
                translate=(0.1, 0.1),     # up to 10% shift
            ),

            # Random perspective transformation
            T.RandomPerspective(distortion_scale=0.2, p=0.5),

            # Random zoom out (adds border and scales image down)
            T.RandomZoomOut(
                side_range=(1.0, 1.5),  # zoom out up to 20%
                p=0.5,
                fill=0
            ),
            
            T.Resize((640, 640)), # Make sure the image is still 640x640 after zoom

            # Color jitter (does NOT affect bboxes)
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),

            # Convert to tensor
            T.ToTensor(),
        ])
    else:
        transforms = T.Compose([
            T.Resize((640, 360)),
            T.Pad(
                padding=(0, 280),
                padding_mode="constant",
                fill=0
            ),
            T.ToTensor(),
        ])
    
    return transforms



# ============================================================
# Dataset
# ============================================================
class BBoxDataset(Dataset):

    def __init__(self, teacher_files, teacher_dir='Teacher', img_dir='Images', transform=None, max_objects = 20):
        self.teacher_files = teacher_files
        self.teacher_dir = teacher_dir
        self.image_dir = img_dir
        self.transform = transform
        self.max_objects = max_objects

    def __len__(self):
        return len(self.teacher_files)

    def __getitem__(self, idx):
        box_file = self.teacher_files[idx]
        img_file = box_file.replace('teacher.json','screen.jpg')
        
        data = json.load(open(os.path.join(self.teacher_dir, box_file), 'r'))
        
        
        
        labels = torch.tensor(data['class'], dtype=torch.int).unsqueeze(1)
        confs = torch.tensor(data['conf'], dtype=torch.float32).unsqueeze(1)

        img_path = os.path.join(self.image_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        
        boxes = data['coords']
        if boxes:
            boxes = BoundingBoxes(
                boxes,
                format="XYXY",
                canvas_size=(data['img_height'], data['img_width'])
                ,dtype=torch.float32
            )
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        
            
        
        if self.transform:
            image, x = self.transform((image, {'boxes':boxes, 'labels':labels, 'confs':confs}))
            
            boxes = x['boxes']
            labels = x['labels']
            confs = x['confs']
        
        
        # Allocate final tensors
        final_boxes = torch.zeros((self.max_objects, 4), dtype=torch.float32)
        final_labels = torch.zeros((self.max_objects, 1), dtype=torch.int)-1
        final_confs = torch.zeros((self.max_objects, 1), dtype=torch.float32)
        
        num_objects = boxes.shape[0]
        
        if num_objects > 0:
            # Flatten conf if needed
            confs_flat = confs.squeeze(-1)
        
            # Sort indices by confidence (descending)
            sorted_idx = torch.argsort(confs_flat, descending=True)
        
            # Reorder tensors
            boxes = boxes[sorted_idx]
            labels = labels[sorted_idx]
            confs = confs[sorted_idx]
        
            # Determine how many we can copy
            k = min(self.max_objects, num_objects)
            
            # Copy top-k
            final_boxes[:k] = boxes[:k]
            final_labels[:k] = labels[:k]
            final_confs[:k] = confs[:k]
            
            
        # Final box is (xc, yc, w, h) normalized
        w = final_boxes[:, 2] - final_boxes[:, 0]
        h = final_boxes[:, 3] - final_boxes[:, 1]
        final_box_centers = torch.stack([
            (final_boxes[:, 0] + w / 2) / 640,
            (final_boxes[:, 1] + h / 2) / 640,
            w / 640,
            h / 640
        ], dim=1)
        
    
        target = {
            "boxes": final_box_centers,
            "labels": final_labels.squeeze(1),
            "confs": final_confs.squeeze(1)
        }

        return image, target




class Head(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()

        # Box branch
        self.box = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_ch, 4, 1)
        )

        # Class branch
        self.cls = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_ch, num_classes, 1)
        )

    def forward(self, x):
        box = self.box(x)
        cls = self.cls(x)
        return torch.cat([box, cls], dim=1)
    
    
    
    
    
class CustomYolov8n(nn.Module):
    def __init__(self, num_classes = 12, backbone = YOLO("yolov8n.pt").model, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.backbone = backbone.to(device)
        in_ch = 144
        
        self.p3_head = Head(in_ch, num_classes).to(device)
        self.p4_head = Head(in_ch, num_classes).to(device)
        self.p5_head = Head(in_ch, num_classes).to(device)
        
    def forward(self, img):
        p3, p4, p5 = self.backbone(img)[1]
        p3 = self.p3_head(p3)
        p4 = self.p4_head(p4)
        p5 = self.p5_head(p5)

        p3 = p3.reshape(p3.shape[0], p3.shape[1], p3.shape[2] * p3.shape[3])
        p4 = p4.reshape(p4.shape[0], p4.shape[1], p4.shape[2] * p4.shape[3])
        p5 = p5.reshape(p5.shape[0], p5.shape[1], p5.shape[2] * p5.shape[3])

        final_head = torch.cat([p3, p4, p5], dim=2)
        return final_head, (p3, p4, p5)
    
    
    

    
class CustomYoloLoss:
    def __init__(self):
        return None
    
    # Convert to x1y1x2y2
    def to_xyxy(self, b):
        return torch.stack([
            b[..., 0] - (b[..., 2] / 2),
            b[..., 1] - (b[..., 3] / 2),
            b[..., 0] + (b[..., 2] / 2),
            b[..., 1] + (b[..., 3] / 2),
        ], dim=-1)
    
    def pairwise_iou(self, pred_boxes, target_boxes, eps=1e-7):
        """
        pred_boxes (B, 4, X) & target_boxes (B, 4, Y)
        returns (B, X, Y) pairwise IoU matrix
        """
        pred = pred_boxes.permute(0, 2, 1) # [B, X, 4]
        target = target_boxes.permute(0, 2, 1) # [B, Y, 4]

        p_xy = self.to_xyxy(pred).unsqueeze(2).expand(-1, -1, target.shape[1], -1)  # [B, X, Y, 4]
        t_xy = self.to_xyxy(target).unsqueeze(1).expand(-1, pred.shape[1], -1, -1)  # [B, X, Y, 4]

        inter_x1 = torch.max(p_xy[..., 0], t_xy[..., 0])
        inter_y1 = torch.max(p_xy[..., 1], t_xy[..., 1])
        inter_x2 = torch.min(p_xy[..., 2], t_xy[..., 2])
        inter_y2 = torch.min(p_xy[..., 3], t_xy[..., 3])
        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        p_area = (p_xy[..., 2] - p_xy[..., 0]) * (p_xy[..., 3] - p_xy[..., 1])
        t_area = (t_xy[..., 2] - t_xy[..., 0]) * (t_xy[..., 3] - t_xy[..., 1])
        union = p_area + t_area - inter

        return inter / (union + eps)  # [B, N, X]
    

    def alignment_score(self, pred_boxes, pred_cls, tgt_boxes, tgt_cls, alpha=0.5, beta=6.0, eps=1e-7):
        """
        pred_boxes: [B, 4, X]
        pred_cls:   [B, C, X]

        tgt_boxes:  [B, 4, Y]
        tgt_cls:    [B, 1] (class indices)
        """

        B, C, X = pred_cls.shape
        Y = tgt_boxes.shape[1]

        valid_gt = (tgt_cls >= 0) # Padded classes with -1

        iou = self.pairwise_iou(pred_boxes, tgt_boxes, eps=eps)  # [B, X, Y]

        # Gather the predicted score for each GT's class at every anchor
        tgt_cls_exp = tgt_cls.clamp(min=0).unsqueeze(1).expand(-1, X, -1).long()  # [B, X, Y]
        cls_scores = pred_cls.permute(0, 2, 1).sigmoid() # [B, X, C]

        # For each anchor n and GT pair, get the predicted score of the GT class
        cls_scores_for_tgt = cls_scores.gather(
            dim=2, index=tgt_cls_exp
        ) # [B, X, Y]

        score = cls_scores_for_tgt ** alpha * iou ** beta # [B, X, Y]
        score = score * valid_gt.unsqueeze(1).float() # zero out invalid GTs

        return score, iou, valid_gt
    

    def select_topk(self, score, valid_gt, k=10):
        # score: [B, X, Y]
        # valid_gt: [B, Y]
        # returns a boolean mask: [B, X, Y]
        # True means this anchor is a candidate for that GT

        # Transpose to [B, Y, X] so topk operates over anchors
        score_t = score.permute(0, 2, 1)               
        num_valid = valid_gt.sum(dim=1).clamp(min=1)
        
        mask = torch.zeros_like(score_t, dtype=torch.bool)
        for b in range(score.shape[0]):
            effective_k = min(k, num_valid[b].item())
            _, idx = score_t[b].topk(effective_k, dim=1)  # [X, k]
            mask[b].scatter_(1, idx, True)

        # Zero out rows belonging to padded GTs
        mask = mask & valid_gt.unsqueeze(2)

        return mask.permute(0, 2, 1)  # [B, X, Y]
    

    def resolve_conflicts(self, mask, score, valid_gt):
        # mask:  [B, X, Y]
        # score: [B, X, Y]

        # If an anchor matches multiple GTs, zero out all but the best
        score_masked = score * mask.float()
        best_gt = score_masked.argmax(dim=2)       # [B, X]

        one_hot = torch.zeros_like(mask)
        one_hot.scatter_(2, best_gt.unsqueeze(2), 1)

        final_mask = one_hot & mask.any(dim=2, keepdim=True)

        # Explicitly exclude padded GTs
        final_mask = final_mask & valid_gt.unsqueeze(1)  # [B, X, Y]

        return final_mask

    
    def TAL(self, pred_boxes, pred_classes, target_boxes, target_classes, alpha=0.5, beta=6.0, k=10, eps=1e-7):
        """
        Task-Aligned Assignment (TAL) algorithm for matching predicted boxes to target boxes.
        This is a simplified version and may not include all the nuances of the original TAL paper.

        pred_boxes: [B, 4, X]
        pred_classes: [B, C, X] (raw logits)
        target_boxes: [B, 4, Y]
        target_classes: [B, 1] (class indices)
        """

        # 1. IoU Matrix & Alignment Scores
        score, iou, valid_gt = self.alignment_score(pred_boxes, pred_classes, target_boxes, target_classes, alpha, beta, eps) # [B, X, Y]

        # 2. Select top-k candidates for each target box
        topk_mask  = self.select_topk(score, valid_gt, k)

        # 3. Resolve conflicts (if a predicted box is assigned to multiple targets, keep the one with highest score)
        final_mask = self.resolve_conflicts(topk_mask, score, valid_gt)

        # 4. Return a final assignment of prediction to target
        #       (functionally a mask that can be applied to the pred and target tensors)
        matched = final_mask.any(dim=2)
        gt_idx = final_mask.float().argmax(dim=2)
        gt_idx = gt_idx * matched.long()
        
        return matched, gt_idx
    
    
    def ciou_loss(self,pred_boxes, target_boxes):
        """
        pred_boxes and target_boxes must both be (B, 4, N) in (xc, yc, w, h) format normalized to [0,1]
        B = Batch size, N = Number of boxes
        """

        pred, target = self.to_xyxy(pred_boxes), self.to_xyxy(target_boxes)
    
        # Intersection
        inter_x1 = torch.max(pred[..., 0], target[..., 0])
        inter_y1 = torch.max(pred[..., 1], target[..., 1])
        inter_x2 = torch.min(pred[..., 2], target[..., 2])
        inter_y2 = torch.min(pred[..., 3], target[..., 3])
        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        
        # Union
        p_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
        t_area = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
        union = p_area + t_area - inter
        iou = inter / (union + 1e-7)
        
        # Enclosing box
        enc_x1 = torch.min(pred[..., 0], target[..., 0])
        enc_y1 = torch.min(pred[..., 1], target[..., 1])
        enc_x2 = torch.max(pred[..., 2], target[..., 2])
        enc_y2 = torch.max(pred[..., 3], target[..., 3])
        enc_diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7
        
        # Center distance penalty
        p_cx = (pred[..., 0] + pred[..., 2]) / 2
        p_cy = (pred[..., 1] + pred[..., 3]) / 2
        t_cx = (target[..., 0] + target[..., 2]) / 2
        t_cy = (target[..., 1] + target[..., 3]) / 2
        center_dist = (p_cx - t_cx) ** 2 + (p_cy - t_cy) ** 2
        
        # Aspect ratio penalty (v term)
        v = (4 / (torch.pi ** 2)) * (
            torch.atan(target[..., 2] / (target[..., 3] + 1e-7)) -
            torch.atan(pred[..., 2] / (pred[..., 3] + 1e-7))
        ) ** 2
        alpha = v / (1 - iou + v + 1e-7)
        
        ciou = iou - (center_dist / enc_diag) - alpha * v
        return (1 - ciou).mean()
