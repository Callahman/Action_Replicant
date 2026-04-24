"""
[WIP] Identification of standard values: health, ammo, and stamina
This is being used to create functions that can quickly identify these values in an image
    without needing a model.

Processes to identify health and ammo are reasonably affective, but further logic
    needs to be added to account for GUI shifting caused by things like equiping/removing backpacks.
"""


# %% Import libraries
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image

import torch.nn as nn
import torchvision.models as models
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from PIL import Image
from Models.enemy_identifier import Enemy_ID_Dataset, EnemyClassifier


# %% Define functions
def get_health(img, health_xs = [220, 590], health_y = 1999):
    
    bar = np.array(img)[health_y-5:health_y+5, 217:595]
    
    health = ((bar[:, :, 0] >= 230) & (bar[:, :, 1] >= 230) & (bar[:, :, 2] >= 230)).astype(np.uint8)
    health_cent = np.sum(health) / np.sum(np.ones(health.shape))
    
    return health_cent


def get_ammo(img, ammo_x = 368, ammo_ys = [1892, 1931]):
    
    clip = np.array(img)[ammo_ys[0]:ammo_ys[1], ammo_x-2:ammo_x+2]
    
    ammo = ((clip[:, :, 0] >= 230) & (clip[:, :, 1] >= 230) & (clip[:, :, 2] >= 230)).astype(np.uint8)
    ammo_cent = np.sum(ammo) / np.sum(np.ones(ammo.shape))
    
    return ammo_cent

preprocess = transforms.Compose([
    transforms.Resize((1080, 1080)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# %% Define globals

data_dir = os.path.abspath('Data')
img_dir = os.path.join(data_dir, 'Images')


# %% itterate over a random set of images and plot the images


# Prep model
model = EnemyClassifier()
if 'Model.pth' in os.listdir():
    model.load_state_dict(torch.load("Model.pth"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


## Itterate over images
df = pd.read_csv('Image_Labels.csv')

temp = df.copy()
temp = df[df['has_enemy'] == 1].sample(frac = 1)
# temp = df[(df['has_enemy'] == 0) & (df['model_label'] >= .75)].sample(frac = 1)
# temp = df[(df['has_enemy'] == 1) & (df['model_label'] <= 0.25)].sample(frac = 1)
# temp = temp.sample(frac = 1).reset_index(drop = True)

files = temp['file'].tolist()[:6]
labels = temp['has_enemy'].tolist()[:6]

# files = np.random.choice(os.listdir(img_dir), 6)

fig, axes = plt.subplots(2, 3)

for ax, file, label in zip(axes.flatten(), files, labels):
    
    img = Image.open(os.path.join(img_dir, file))
    
    ax.imshow(img)
    
    
    # Create and plot heatmap
    orig_w, orig_h = img.size
    
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    # yhat, heatmap = gradcam(input_tensor)
    yhat, heatmap = model.heatmap(input_tensor, target_size=(orig_h, orig_w))
    ax.imshow(heatmap.cpu(), cmap='jet', alpha=0.25)
    
    
    cent_health = round(get_health(img)*100, 2)
    cent_ammo = round(get_ammo(img)*100, 2)
    if yhat >= .5:
        enemy = 'Has Enemy: Yes'
    else:
        enemy = 'Has Enemy: No'
    ax.set_title(f'{enemy}     Label: {label}     Health: {cent_health}%     Ammo: {cent_ammo}%')
    ax.axis('off')

plt.tight_layout(pad=0)
plt.show()
