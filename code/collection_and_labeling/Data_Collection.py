"""
When executed, this script captures screenshots, keyboard inputs, and mouse movements/inputs

The keyboard/mouse data is collected prior to the screenshot
Filenames for inputs and screenshots are determined by the timestamp of the screenshot/event-bundles
This is done so marrying the data later for training/analysis can be streamlined
"""


# %% Import Libraries
import pyautogui

import numpy as np
import cv2
from mss import mss
import os

from pynput import mouse, keyboard
from time import time, sleep, gmtime, strftime

import pickle

from collections import deque
from threading import Thread
from queue import Queue

# Import paths (for directory management)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # up 2 levels → Main_Dir
import paths



# %% Create Necessary Directories for Data Collection
img_dir = paths.IMAGE_DIR
pickle_dir = paths.PICKLE_DIR

# %% Get the window bounding box for the game
x = pyautogui.getAllWindows()

# Default to HELLDIVERS as that is the focus of this repo, could be changed for other games/applications
target = "HELLDIVERS"
for y in x:
    title = y.title
    
    print(title)
    if not target in title:
        continue
    
    window_bar = 50
    
    left = y.left
    top = y.top
    width = y.width
    height = y.height
    break



# Screen capture window
bounding_box = {'top': top
                , 'left': left
                , 'width': width
                , 'height': height}
sct = mss()




# %% Create queue to save data on a different thread
save_queue = Queue()

def save_worker():
    while True:
        img, img_id, data = save_queue.get()
        if img is None:
            break
        cv2.imwrite(f'{img_dir}/{img_id}_screen.jpg', img)
        pickle.dump(data, open(f'{pickle_dir}/{img_id}.pickle','wb'))
        save_queue.task_done()


# %% Setup tracking for mouse and keyboard events
event_data = {'keyboard': deque(), 'mouse_moves': deque(), 'mouse_clicks': deque()}

def on_key_press(key):
    event_data['keyboard'].append({'time':time(),'key':str(key),'pressed':True})


def on_key_release(key):
    event_data['keyboard'].append({'time':time(), 'key':str(key),'pressed':False})


def on_move(x, y):
    event_data['mouse_moves'].append({'time':time(), 'x':x, 'y':y})


def on_click(x, y, button, pressed):
    event_data['mouse_clicks'].append({'time':time(), 'x': x, 'y': y, 'button': str(button), 'pressed': pressed})


def save_data():
    data_copy = {}
    for k, v in event_data.items():
        data_copy[k] = list(v)
        v.clear()
    return data_copy






# %% Main loop to capture screenshots and save data
if __name__ == "__main__":
    
    
    # Provide a countdown after initial execution (gives time to switch back to the game)
    count_until = 5
    for x in range(count_until):
        print(f'Starting in {count_until-x}...')
        pyautogui.sleep(1)
    print('GO!!!\n')
    
    
    # Start thread to save data (keeps the pipeline clear to grab images/data)
    Thread(target=save_worker, daemon=True).start()
    
    # Create listener for keyboard events
    keyboard_listener = keyboard.Listener(on_press=on_key_press, on_release = on_key_release)
    keyboard_listener.start()
    
    # Create listener for mouse events
    mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)
    mouse_listener.start()
    
    
    
    size = 1
    x_resize = int(round(width / size, 0))
    y_resize = int(round(height / size, 0))
    
    
    target_fps = 10
    frame_time = 1.0 / target_fps
    last_frame = time()
    
    start_time = time()
    img_count = len(os.listdir(img_dir))
    while True:
        
        current_time = time()
        img_id = strftime('%Y_%m_%d_%H_%M_%S', gmtime(current_time))
        microseconds = int((current_time % 1) * 1e6)
        img_id += f'_{microseconds:06d}'
        
    
        ### Resize Main Image
        img = np.array(sct.grab(bounding_box))[:,:,:3]
        img = cv2.resize(img, (x_resize, y_resize))
        
        
        # Package the latest batch of data
        x = save_data()
        
        
        
        # Save data
        save_queue.put((img, img_id, x))
        
        
        now = time()
        if now - last_frame < frame_time:
            sleep(frame_time - (now - last_frame))
        last_frame = time()
        
        img_count += 1
        if img_count % 100 == 0:
            print(f'{img_count} images so far')


