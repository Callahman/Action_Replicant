"""
This script takes the result of "Image_BBox_Labeler.py" (bounding boxes without class labels)
    and adds class labels to the bounding boxes.
The classes are predefined in the code for Helldivers II, and strictly focuses on enemies.

Steps:
    1. Run the script
    2. Using the displayed images, identify the class
        - One image shows the full screenshot
        - The other image shows the area covered by the bounding box to help with identification
    3. In your terminal, input the number corresponding to the class and press "Enter"
    4. Repeat steps 2-3 for each image in the labeled (bbox only) dataset

Future Labeling Improvements:
    - Add process to dynamically add classes if new classes appear in images
    - Change from a manual key press, to a dropdown menu or clickable GUI
        - This would likely speed up the process and allow for more classes without increasing the likelihood of errors
    - Add classes for some combination of:
        - Enemy type, rather than size + type
            - Ex: large_bot would become "tank" or "hulk", which would carry more speceficity later
            - Adds risk of mislabeling larger enemies as smaller ones


Note: This should probably be switch to two processes:
    1. Adding the class labels to the bounding boxes
    2. Converting the now labeled (bbox + class) data into COCO format
"""

import pickle
import os
import json
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
import shutil
import numpy as np


data_dir = os.path.abspath('Data')
img_dir = os.path.join(data_dir, 'Images')
box_dir = os.path.join(data_dir, 'BBoxes')


dataset_dir = os.path.join(data_dir, 'dataset')
train_dir = os.path.join(dataset_dir, 'train')
valid_dir = os.path.join(dataset_dir, 'valid')
test_dir = os.path.join(dataset_dir, 'test')

os.mkdir(dataset_dir, exists_ok=True)
os.mkdir(train_dir, exists_ok=True)
os.mkdir(valid_dir, exists_ok=True)
os.mkdir(test_dir, exists_ok=True)



# Define classes
sizes = ["small", "medium", "large", "massive"]
types = ["bot", "bug", "squid"]
classes = [f"{s}_{t}" for s in sizes for t in types]  # 12 total

coco_output = {
    "info": {"description": "Manual bounding box classification"},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [
        {"id": i + 1, "name": c, "supercategory": c.split("_")[1]} for i, c in enumerate(classes)
    ]
}


if 'COCO_Outputs.pickle' not in os.listdir():
    outputs = {
        'train':{'output':deepcopy(coco_output), 'annotation_id':1, 'img_idx':0},
        'valid':{'output':deepcopy(coco_output), 'annotation_id':1, 'img_idx':0},
        'test':{'output':deepcopy(coco_output), 'annotation_id':1, 'img_idx':0}
        }
    completed = []
else:
    outputs = pickle.load(open('COCO_Outputs.pickle','rb'))
    completed = list(set(outputs['completed']))




# Setup files
box_files = [x for x in os.listdir(box_dir) if x.replace('_bbox.pickle','_screen.jpg') not in completed]
random_i = np.random.randint(0, len(box_files)-1)
box_files = box_files[random_i:] + box_files[:random_i]

# Create figure ONCE (persistent across bboxes)
plt.style.use("dark_background")
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
plt.tight_layout(pad=0)

plt.ion()  # turn on interactive mode
fig.show()

# Establish "Train" as the first set to get labels
cent_remaining = 1-(len(completed) / len(os.listdir(box_dir)))

set_type = 'train'
set_count = 0
print(f'Percent Remaining: {round(cent_remaining * 100, 1)}%')
print('Labeling Train Set...')

for box_file in box_files:
    img_file = box_file.replace('_bbox.pickle', '_screen.jpg')
    new_img = os.path.join(dataset_dir, set_type, img_file)
    
    if img_file in completed:
        continue

    image = cv2.cvtColor(cv2.imread(os.path.join(img_dir, img_file)), cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    bboxes = pickle.load(open(os.path.join(box_dir, box_file), 'rb'))

    output = outputs[set_type]['output']
    annotation_id = outputs[set_type]['annotation_id']
    img_idx = outputs[set_type]['img_idx']

    output["images"].append({
        "id": img_idx + 1,
        "file_name": img_file,
        "height": height,
        "width": width
    })

    for bbox_idx, (x1, y1, x2, y2) in enumerate(bboxes):
        # Denormalize bbox
        x1 = int(round(float(x1) * width, 0))
        y1 = int(round(float(y1) * height, 0))
        x2 = int(round(float(x2) * width, 0))
        y2 = int(round(float(y2) * height, 0))
        w, h = x2 - x1, y2 - y1

        pad = 10
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(width, x2 + pad), min(height, y2 + pad)

        # Update existing figure instead of recreating
        axs[0].clear()
        axs[1].clear()
        
        axs[0].axis('off')
        axs[1].axis('off')

        axs[0].imshow(image)
        axs[0].add_patch(plt.Rectangle((x1, y1), w, h, fill=False, color='red', linewidth=2))
        axs[0].set_title(f"Image {img_idx + 1}")

        axs[1].imshow(image[y1:y2, x1:x2])
        axs[1].set_title("Region around bbox")

        fig.suptitle(
            "Classes:\n" + "\n".join([f"{i}: {name}" for i, name in enumerate(classes)]),
            fontsize=10,
        )

        fig.canvas.draw()
        fig.canvas.flush_events()

        # User input
        while True:
            try:
                cls_id = int(input(f"({len(os.listdir(box_dir)) - len(completed)}) Enter class number (0–{len(classes) - 1}) for this bbox: "))
                if 0 <= cls_id < len(classes):
                    break
                else:
                    print("Invalid number, try again.")
            except ValueError:
                print("Please enter a valid integer.")

        # Add annotation
        output["annotations"].append({
            "id": annotation_id,
            "image_id": img_idx + 1,
            "category_id": cls_id + 1,
            "bbox": [x1, y1, w, h],
            "area": w * h,
            "iscrowd": 0
        })
        annotation_id += 1
    
    # Save files [copy img and save output file(s)]
    shutil.copy(os.path.join(img_dir, img_file), new_img)
    completed.append(img_file)
    
    if len(completed) % 100 == 0:
        cent_remaining = 1-(len(completed) / len(os.listdir(box_dir)))
        print(f'\n\nPercent Remaining: {round(cent_remaining * 100, 3)}%\n\n')
    
    outputs[set_type]['output'] = deepcopy(output)
    outputs[set_type]['annotation_id'] = annotation_id
    outputs[set_type]['img_idx'] = img_idx+1
    outputs['completed'] = deepcopy(completed)
    pickle.dump(outputs, open('COCO_Outputs.pickle','wb'))
    
    output_path = os.path.join(dataset_dir, set_type, "_annotations.coco.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    
    # Update the set type if needed
    set_count += 1
    if set_type == 'train' and set_count >= 8:
        print('Labeling Validation Set...')
        set_count = 0
        set_type = 'valid'
    elif set_type == 'valid' and set_count >= 2:
        print('Labeling Test Set...')
        set_count = 0
        set_type = 'test'
    elif set_type == 'test' and set_count >= 2:
        print('Labeling Train Set...')
        set_count = 0
        set_type = 'train'

plt.ioff()  # turn off interactive mode
plt.close(fig)
