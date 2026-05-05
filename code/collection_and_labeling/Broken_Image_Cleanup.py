"""
This iterates over the images you've collected and attempts to open each one.
If an image is corrupted or otherwise cannot be opened, the image and all associated files are deleted.

Current design minimizes compute resources, but takes a long time to run.
Since this is a one-time cleanup after collection I didn't worry about efficiency.

Future Improvements:
    - Parrallelize the process to speed it up (e.g., using multiprocessing or concurrent.futures)
    - Implement a more robust logging system to track which files were removed and why (e.g., using the logging module)
    - Add a backup mechanism before deletion in case of accidental removals (e.g., move files to a "quarantine" folder instead of deleting immediately)
"""


from PIL import Image
import os
from tqdm import tqdm


# Import paths (for directory management)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # up 2 levels → Main_Dir
import paths


img_dir = paths.IMAGE_DIR
bbox_dir = paths.BBOX_DIR
pickle_dir = paths.PICKLE_DIR
teacher_dir = paths.TEACHER_DATA_DIR
collection_dir = paths.COLLECT_DIR


if 'validated_images.txt' in os.listdir(collection_dir):
    with open(os.path.join(collection_dir, 'validated_images.txt'), 'r') as f:
        validated_images = list(set(line.strip() for line in f))

else:
    validated_images = []

images = list(set(os.listdir(img_dir)) - set(validated_images))
# images = [img for img in os.listdir(img_dir) if img not in validated_images]

bbox_list = os.listdir(bbox_dir)
pickle_list = os.listdir(pickle_dir)
teacher_list = os.listdir(teacher_dir)
image_list = os.listdir(img_dir)


removals = 0
for image_file in tqdm(images):
    img_path = os.path.join(img_dir, image_file)

    bbox = image_file.replace('screen.jpg', 'bbox.pickle')
    pickle = image_file.replace('_screen.jpg', '.pickle')
    teacher = image_file.replace('_screen.jpg', '_teacher.json')

    try:
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        validated_images.append(image_file)

    except KeyboardInterrupt:
        print("Process interrupted by user. Exiting.")
        break

    except OSError:
        box_check = bbox in bbox_list
        pickle_check = pickle in pickle_list
        teacher_check = teacher in teacher_list
        img_check = image_file in image_list

        if box_check:
            os.remove(os.path.join(bbox_dir, bbox))
        if pickle_check:
            os.remove(os.path.join(pickle_dir, pickle))
        if teacher_check:
            os.remove(os.path.join(teacher_dir, teacher))

        os.remove(os.path.join(img_dir, image_file))

        removals += 1


with open(os.path.join(collection_dir, 'validated_images.txt'), 'w') as f:
    img_text = '\n'.join(validated_images)
    f.write(img_text)

print(f"Total broken images identified and cleaned: {removals}")