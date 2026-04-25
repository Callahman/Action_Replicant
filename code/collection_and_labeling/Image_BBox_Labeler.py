"""
Process to create bounding box labels for each image in the dataset.
Includes options use of a pretrained model to perform an initial "guess".
If the guess is acceptable, the user can accept the labels and move forward.*
BBoxes alone are covered in this process.
    - This was done to simplify the requirements for the user and speed up this labeling step

Steps to label:
1. Run the script
2. If using the pretrained model, determine if the boxes are correct.
    - If so, press "Enter" to accept the model's labels
    - If not, continue to step 3
3. Left click on the image to indicate the 1st corner of the bounding box,
    then left click again to indicate the opposite corner of the bounding box.
4. Repeat step 3 for each enemy in the image.
5. Press "Enter" to save the labels and move to the next image.
    - If there were no enemies, the process automatically jumps to a random image
    - If there are enemies, the process moves to the next image in order to maximize enemy capture
        - Hitting the "Space" key will break this sequence
            and jump to a random image ignoring any labels on the current image.
6. Repeat steps 2-5 until:
    - Exhausted with labeling images
    - All images are labeled 🎉🎉🎉

* The model also helps to highlight enemies that would otherwise be missed when manually labeling. 
Helpful for the model to spot the enemies and the the user refines the output

Future improvements:
- Add COCO formating output so it doesn't need to be converted later for training
- Add process to adjust boxes rather than having to redraw them if the model's guess needs refining
"""



# %% Import libraries
import os
import numpy as np
import pickle

import matplotlib.pyplot as plt
from PIL import Image

from typing import List, Tuple

from rfdetr import RFDETRBase
import torch

# Import paths for directory management
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # up 2 levels → Main_Dir
import paths


# %% Define globals
# IF PRETRAINED HELPER MODEL IS AVAILABLE
USE_PRETRAINED_MODEL = True

img_dir = paths.IMAGE_DIR
box_dir = paths.BBOX_DIR
dataset_dir = paths.DATASET_DIR
output_dir = paths.OUTPUT_DIR


if USE_PRETRAINED_MODEL:
    model = RFDETRBase(pretrain_weights=os.path.join(output_dir, 'checkpoint_best_ema.pth'), device='cuda')
    model.model.model = torch.compile(model.model.model, backend="aot_eager")


# Pre-determine the file list
img_files = np.array(os.listdir(img_dir))
bbox_files = np.array(os.listdir(box_dir))

bbox_bases = np.char.replace(bbox_files, 'bbox.pickle', 'screen.jpg')
mask = ~np.isin(img_files, bbox_bases)
files = img_files[mask].tolist()



plt.ion()

fig, ax = plt.subplots()
ax.axis('off')

plt.tight_layout(pad=0)


# %% Define functions
def sort_bboxes_bottom_right_first(bboxes: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
    # Sort by y2 descending, then x2 descending
    sorted_bboxes = sorted(
        bboxes,
        key=lambda box: (box[3], box[2]),  # (y2, x2)
        reverse=True  # descending order
    )
    return sorted_bboxes

current_idx = np.random.randint(0, len(files))
completed_idx = []
clicks = []  # store current bounding box clicks
boxes = []   # store all boxes for current image
rects = []   # matplotlib rectangles drawn

def show_image(idx, model, USE_PRETRAINED_MODEL):
    """Display image at dataframe index idx."""
    ax.clear()
    ax.axis('off')
    file = files[idx]
    img = Image.open(os.path.join(img_dir, file))
    ax.imshow(img)
    
    # Clear any old patches when showing new image
    for p in list(ax.patches):
        p.remove()
    
    if USE_PRETRAINED_MODEL:
        preds = model.predict(os.path.join(img_dir, file))
        if len(preds) > 0:
            
            for p in preds:
                
                coords = p[0]
                
                x1 = coords[0]
                y1 = coords[1]
                x2 = coords[2]
                y2 = coords[3]
                boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
                
                rect = plt.Rectangle(
                    (min(x1, x2), min(y1, y2)),
                    abs(x2 - x1),
                    abs(y2 - y1),
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none'
                )
                ax.add_patch(rect)

    fig.canvas.draw()
    
    
def draw_boxes():
    """Visualize all drawn bounding boxes safely."""
    # Clear existing rectangles safely
    for p in list(ax.patches):
        p.remove()

    # Draw new rectangles
    for (x1, y1, x2, y2) in boxes:
        rect = plt.Rectangle(
            (min(x1, x2), min(y1, y2)),
            abs(x2 - x1),
            abs(y2 - y1),
            linewidth=2,
            edgecolor='lime',
            facecolor='none'
        )
        ax.add_patch(rect)

    fig.canvas.draw_idle()
    
    
def on_click(event):
    global clicks, boxes

    if event.button == 1 and event.inaxes:  # Left click
        clicks.append((event.xdata, event.ydata))
        if len(clicks) == 2:
            boxes.append((*clicks[0], *clicks[1]))
            clicks = []
            draw_boxes()

    elif event.button == 3 and event.inaxes:  # Right click = reset
        clicks = []
        boxes = []
        draw_boxes()


def on_key(event, model, USE_PRETRAINED_MODEL):
    """Handle key press for labeling."""
    global current_idx, boxes

    if event.key == 'enter':
        img_file = files[current_idx]
        bbox_file = img_file.replace('screen.jpg','bbox.pickle')
        
        if len(boxes) > 1:
            boxes = sort_bboxes_bottom_right_first(boxes)
            
        # Load the image to get its dimensions for normalization
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path)
        img_w, img_h = img.size

        # Normalize coordinates by image width/height
        boxes = [
            (
                min(x1, x2) / img_w,
                min(y1, y2) / img_h,
                max(x1, x2) / img_w,
                max(y1, y2) / img_h
                )
            for (x1, y1, x2, y2) in boxes
            ]
        
        pickle.dump(boxes, open(os.path.join(box_dir, bbox_file), 'wb'))

        # Move to next image
        completed_idx.append(current_idx)
        if boxes and current_idx + 1 not in completed_idx and current_idx < len(files)-1:
            current_idx += 1
        else:
            options = list(set(list(range(len(files)))) - set(completed_idx))
            if options:
                current_idx = np.random.choice(options)
            else:
                current_idx = -1
        
        
        if len(completed_idx) < len(files) and current_idx != -1:
            boxes = []
            clicks = []
            show_image(current_idx, model, USE_PRETRAINED_MODEL)
        else:
            print("🎉 All images labeled!")
            plt.close(fig)
            
    if event.key == 'space':
        options = list(set(list(range(len(files)))) - set(completed_idx))
        if options:
            current_idx = np.random.choice(options)
        else:
            current_idx = -1
    
    
        if current_idx != -1:
            boxes = []
            clicks = []
            show_image(current_idx, model, USE_PRETRAINED_MODEL)
        else:
            print("🎉 All images labeled!")
            plt.close(fig)




# Connect the key press handler
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', lambda event: on_key(event, model, USE_PRETRAINED_MODEL))

# Start labeling
if files:
    show_image(current_idx, model, USE_PRETRAINED_MODEL)
    plt.show(block=True)
else:
    print("✅ No images found.")
