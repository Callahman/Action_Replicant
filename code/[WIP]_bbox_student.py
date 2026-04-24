
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



# =============================================================================
# """
# Questions about downloading/saving pytorch yolo for later:
#     - Save version that works with different image size?
#     - You say the target tensor is shape (N, 6), but then say
#         [class, x_center, y_center, width, height], where this is len(5)
# """
# 
# from ultralytics import YOLO
# import torch
# 
# yolo = YOLO("yolov8n.pt")
# 
# torch_model = yolo.model  # <-- nn.Module
# torch_model.eval()
# 
# 
# torch.save(torch_model.state_dict(), "yolov8n_backbone.pt")
# =============================================================================



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
# =============================================================================
#         w = final_boxes[:, 2] - final_boxes[:, 0]
#         h = final_boxes[:, 3] - final_boxes[:, 1]
#         final_box_centers = torch.stack([
#             (final_boxes[:, 0] + w / 2) / 640,
#             (final_boxes[:, 1] + h / 2) / 640,
#             w / 640,
#             h / 640
#         ], dim=1)
# =============================================================================
        
    
        target = {
            "boxes": final_boxes,
            "labels": final_labels,
            "confs": final_confs
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
    def __init__(self, num_classes = 12, backbone = YOLO("yolov8n.pt").model):
        super().__init__()
        
        self.backbone = backbone
        in_ch = 144
        
        self.p3_head = Head(in_ch, num_classes)
        self.p4_head = Head(in_ch, num_classes)
        self.p5_head = Head(in_ch, num_classes)
        
    def forward(self, img):
        p3, p4, p5 = self.backbone(img)
        return self.p3_head(p3), self.p4_head(p4), self.p5_head(p5)
    
    
    

    
class CustomYoloLoss:
    def __init__(self):
        return None
    
    def assignment(self, bbox, label, p3_threshold = 32**2, p4_threshold = 96**2):
        area = (bbox[:, :, 2] - bbox[:, :, 0]) * (bbox[:, :, 3] - bbox[:, :, 1])
        
        p3 = torch.zeros(bbox.shape, dtype = torch.float32, device = bbox.device)
        p4 = torch.zeros(bbox.shape, dtype = torch.float32, device = bbox.device)
        p5 = torch.zeros(bbox.shape, dtype = torch.float32, device = bbox.device)
        
        p3[area < p3_threshold] = bbox[area < p3_threshold]
        p4[(p3_threshold < area) & (area < p4_threshold)] = bbox[(p3_threshold < area) & (area < p4_threshold)]
        p5[area >= p4_threshold] = bbox[area >= p4_threshold]
        
        
        p3_label = torch.zeros(label.shape, dtype = torch.int, device = label.device)-1
        p4_label = torch.zeros(label.shape, dtype = torch.int, device = label.device)-1
        p5_label = torch.zeros(label.shape, dtype = torch.int, device = label.device)-1
        
        p3_label[area < p3_threshold] = label[area < p3_threshold]
        p4_label[(p3_threshold < area) & (area < p4_threshold)] = label[(p3_threshold < area) & (area < p4_threshold)]
        p5_label[area >= p4_threshold] = label[area >= p4_threshold]
        
        p3_mask = (p3_label.squeeze(-1) >= 0).float()
        p4_mask = (p4_label.squeeze(-1) >= 0).float()
        p5_mask = (p5_label.squeeze(-1) >= 0).float()
        
        return p3, p3_label, p3_mask, p4, p4_label, p4_mask, p5, p5_label, p5_mask
    
    
    def decode(self, pred):
        
        stride = 640 / pred.shape[-1]
        box = pred[:, :4].permute(0, 2, 3, 1)
        labels = pred[:, 4:]
        
        N, H, W, _ = box.shape
        device = box.device
        
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        
        grid = torch.stack((x, y), dim=-1).float()  # (H, W, 2)
        
        center = (grid + 0.5) * stride
        
        l = box[..., 0]
        r = box[..., 1]
        t = box[..., 2]
        b = box[..., 3]
        
        cx = center[..., 0]
        cy = center[..., 1]
        
        x1 = cx - l
        y1 = cy - t
        x2 = cx + r
        y2 = cy + b
        
        decoded = torch.stack([x1, y1, x2, y2], dim=-1)  # (N, H, W, 4)
        pred_bbox = decoded.reshape(N, -1, 4)
        
        pred_labels = labels.reshape(N, -1, labels.shape[1])
        
        return pred_bbox, pred_labels
    
    
    def pairwise_ciou(self, pred, gt, eps=1e-7):
        # pred: (N, X, 4)
        # gt:   (N, Y, 4)
        N, X, _ = pred.shape
        _, Y, _ = gt.shape
    
        pred_exp = pred[:, :, None, :]  # (N, X, 1, 4)
        gt_exp   = gt[:, None, :, :]    # (N, 1, Y, 4)
    
        # Intersection
        tl = torch.max(pred_exp[..., :2], gt_exp[..., :2])
        br = torch.min(pred_exp[..., 2:], gt_exp[..., 2:])
        wh = (br - tl).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
    
        # Union
        area_pred = (pred[..., 2]-pred[..., 0]) * (pred[..., 3]-pred[..., 1])
        area_gt   = (gt[..., 2]-gt[..., 0]) * (gt[..., 3]-gt[..., 1])
        union = area_pred[:, :, None] + area_gt[:, None, :] - inter
        iou = inter / (union + eps)
    
        # Center distance
        px = (pred_exp[..., 0] + pred_exp[..., 2]) / 2
        py = (pred_exp[..., 1] + pred_exp[..., 3]) / 2
        gx = (gt_exp[..., 0] + gt_exp[..., 2]) / 2
        gy = (gt_exp[..., 1] + gt_exp[..., 3]) / 2
        rho2 = (px - gx) ** 2 + (py - gy) ** 2
    
        # Enclosing box diagonal
        c_tl = torch.min(pred_exp[..., :2], gt_exp[..., :2])
        c_br = torch.max(pred_exp[..., 2:], gt_exp[..., 2:])
        c2 = ((c_br[..., 0] - c_tl[..., 0])**2 + (c_br[..., 1] - c_tl[..., 1])**2).clamp(min=eps)
    
        # Aspect ratio
        pw = (pred_exp[..., 2] - pred_exp[..., 0]).clamp(min=eps)
        ph = (pred_exp[..., 3] - pred_exp[..., 1]).clamp(min=eps)
        gw = (gt_exp[..., 2] - gt_exp[..., 0])
        gh = (gt_exp[..., 3] - gt_exp[..., 1])
        v = (4 / (math.pi**2)) * (torch.atan(gw/gh) - torch.atan(pw/ph))**2
        alpha = v / (1 - iou + v + eps)
    
        ciou = iou - rho2 / c2 - alpha * v
        return ciou  # (N, X, Y)

