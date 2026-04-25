"""
Process to visually evaluate the performance of a trained RFDETR model on the dataset.

Compares the predicted bounding boxes/labels against the ground-truth boxes/labels
"""


import os
import pickle
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from rfdetr import RFDETRBase
from itertools import product

# Import paths (for directory management)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # up 2 levels → Main_Dir
import paths





def draw_dotted_rectangle(img, top_left, bottom_right, color=(255, 255, 255), thickness=2, gap=10):
    """
    Draw a dotted rectangle on img.
    
    Parameters:
        img - image to draw on
        top_left - (x1, y1)
        bottom_right - (x2, y2)
        color - BGR color tuple
        thickness - line thickness
        gap - length of the gap between dots
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Top edge
    for x in range(x1, x2, gap*2):
        cv2.line(img, (x, y1), (min(x+gap, x2), y1), color, thickness)
    # Bottom edge
    for x in range(x1, x2, gap*2):
        cv2.line(img, (x, y2), (min(x+gap, x2), y2), color, thickness)
    # Left edge
    for y in range(y1, y2, gap*2):
        cv2.line(img, (x1, y), (x1, min(y+gap, y2)), color, thickness)
    # Right edge
    for y in range(y1, y2, gap*2):
        cv2.line(img, (x2, y), (x2, min(y+gap, y2)), color, thickness)



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


# ---------------- COLORS ----------------
def generate_patch_colors(num_patches):
    cmap = plt.cm.get_cmap("tab20", num_patches)
    return [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(num_patches)
    ]


# ---------------- SAFETY ----------------
def ensure_imshow_shape(img):
    """
    Guarantees (H, W, 3) for matplotlib.imshow
    """
    if img.ndim == 4:
        img = img[0]
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Invalid image shape for imshow: {img.shape}")
    return img


# ---------------- INFERENCE ----------------
@torch.no_grad()
def run_inference_on_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"OpenCV could not read image: {image_path}")

    # Handle alpha channel if present
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tiles = tile_image(image, PATCH_SIZE)
    colors = generate_patch_colors(len(tiles))
    
    all_preds = []
    offsets = []
    for i, t in enumerate(tiles):
        patch = t[0]
        offset = (t[1], t[2])
        
        print(f'Predicting Patch: {i + 1} of {len(tiles)}')
        preds = model.predict(patch, conf_threshold=0.3, device=DEVICE)
        
        if len(preds) == 0:
            continue
        all_preds.append(preds)
        offsets.append(offset)

    all_detections = []
    patch_h, patch_w = patch.shape[:2]
    for preds, (offset_x, offset_y), color in zip(all_preds, offsets, colors):
        for det in preds:
            
            x1 = det[0][0] + offset_x
            y1 = det[0][1] + offset_y
            x2 = det[0][2] + offset_x
            y2 = det[0][3] + offset_y
            all_detections.append({
                "box": (x1, y1, x2, y2),
                "class_id": det[3],
                "score": det[2],
                "color": color
            })

    return image, all_detections, tiles


# ---------------- DRAW ----------------
def draw_detections(image, detections, class_names, tiles=None):
    vis = image.copy()

    # Draw patch borders first
    if tiles is not None:
        for _, x_offset, y_offset in tiles:
            top_left = (x_offset, y_offset)
            bottom_right = (x_offset + PATCH_SIZE, y_offset + PATCH_SIZE)
            draw_dotted_rectangle(vis, top_left, bottom_right, color=(255,255,255), thickness=2, gap=15)

    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        class_name = class_names[det["class_id"]]
        score = det["score"]
        color = det["color"]

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 4)
        cv2.putText(
            vis,
            f"{class_name} {score:.2f}",
            (x1, max(y1 - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return vis


# ---------------- DRAW GROUND-TRUTH + PREDICTIONS ----------------
def draw_boxes_comparison(image, gt_boxes, pred_detections, class_names, gt_color=(0, 255, 0), pred_color=(255, 0, 0)):
    """
    Draw ground-truth boxes and predicted boxes on the image.
    
    Parameters:
        image: np.array, RGB
        gt_boxes: list of [x1, y1, x2, y2], normalized 0-1 coordinates
        pred_detections: list of dicts with "box" key in absolute coordinates
        gt_color: color for ground-truth boxes (default green)
        pred_color: color for predicted boxes (default red)
    """
    vis = image.copy()
    h, w = vis.shape[:2]

    # Draw ground-truth boxes
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        # scale normalized coordinates to image size
        x1, y1 = int(x1 * w), int(y1 * h)
        x2, y2 = int(x2 * w), int(y2 * h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), gt_color, 3)
        cv2.putText(vis, "GT", (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 2)

    # Draw predicted boxes
    for det in pred_detections:
        x1, y1, x2, y2 = map(int, det["box"])
        class_name = class_names[det["class_id"]]
        score = det.get("score", 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), pred_color, 3)
        cv2.putText(vis, f"P {score:.2f}     |     Class: {class_name}", (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 2)
        
        print(class_name, det['class_id'])

    return vis


# ---------------- CONFIG ----------------
PATCH_SIZE = 896
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = 'cpu'

checkpoint_path = "Patch_Data/dataset/output/checkpoint.pth"

# image_dir = paths.IMAGE_DIR
image_dir = paths.VALID_DIR

# =============================================================================
# sizes = ["small", "medium", "large", "massive"]
# types = ["bot", "bug", "squid"]
# class_names = [f"{s}_{t}" for s in sizes for t in types]
# =============================================================================

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
    num_classes=len(class_names),
    resolution=PATCH_SIZE,
    pretrain_weights=checkpoint_path,
    device=DEVICE
)
model.model.model.eval()

model.model.model = torch.compile(
    model.model.model,
    backend="aot_eager"
)


# ---------------- LOOP FOR PATCH PERFORMANCE ----------------
# =============================================================================
# bbox_dir = paths.BBOX_DIR
# box_files = os.listdir(bbox_dir)
# np.random.shuffle(box_files)
# 
# 
# plt.ion()
# fig, ax = plt.subplots(figsize=(14, 8))
# im = None
# ax.axis("off")
# for fname in box_files:
#     
#     x = pickle.load(open(os.path.join(bbox_dir, fname), 'rb'))
#     if not x:
#         continue
#     
#     fname = fname.replace('bbox.pickle','screen.jpg')
# 
#     img_path = os.path.join(image_dir, fname)
#     # print(f"Inferencing: {img_path}")
#     image, detections, tiles = run_inference_on_image(img_path)
#     # break
#     vis = draw_detections(image, detections, tiles=tiles)
#     vis = ensure_imshow_shape(vis)
# 
#     if im is None:
#         im = ax.imshow(vis)
#     else:
#         im.set_data(vis)
# 
#     ax.set_title(fname)
#     fig.canvas.draw_idle()
#     plt.pause(1)
# 
# 
# plt.ioff()
# plt.show()
# =============================================================================



# ---------------- LOOP FOR VISUALIZATION ----------------
plt.ion()
dpi = 200
fig_width = 3842 / dpi
fig_height = 2162 / dpi

fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
im = None
ax.axis("off")

bbox_dir = paths.BBOX_DIR
# box_files = os.listdir(bbox_dir)
# np.random.shuffle(box_files)

img_files = os.listdir(image_dir)
np.random.shuffle(img_files)

for img_fname in img_files:
# for fname in box_files:
    
    fname = img_fname.replace('screen.jpg','bbox.pickle')
    
    bbox_file = fname
    x = pickle.load(open(os.path.join(bbox_dir, fname), 'rb'))  # list of normalized GT boxes
    if not x:
        continue
    
    # img_fname = fname.replace('bbox.pickle','screen.jpg')
    img_path = os.path.join(image_dir, img_fname)

    # Run inference
    print(f'Inference: {fname}')
    image, detections, _ = run_inference_on_image(img_path)  # tiles=None
    # Draw GT and predictions
    vis = draw_boxes_comparison(image, gt_boxes=x, pred_detections=detections, class_names=class_names)

    vis = ensure_imshow_shape(vis)
    if im is None:
        im = ax.imshow(vis)
    else:
        im.set_data(vis)

    ax.set_title(img_fname)
    fig.canvas.draw_idle()
    plt.pause(3)

plt.ioff()
plt.show()
