# Code Overview
## Data Collection and Labeling
- Step 1: Data collection is handled by Data_Collection.py
    - This script uses a hard-coded application name (ex: HelldiversII) to collect date at runtime

- Step 2: Initial labeling is handled by Image_BBox_Labeler.py
    - This script is designed to create the first round of labels, adding BoundingBoxes for relevant targets

- Step 3: Secondary labeling is handled by Image_COCO_Labeler.py
    - This script takes the BoundingBoxes and adds a class label. It also outputs the data into COCO-style for RFDETR training
    - May be better to split this script into 2 scripts: class labeling & COCO-style formatting


## Teacher
- Step 4: Create a secondary dataset for Teacher Training with Create_PatchTeacher_Dataset.py
    - The secondary dataset represents the core image + bbox dataset converted into patches.
        - Patches allow the high-resolution of the original image to be maintained while converting the image to a size native for RFDETR
        - Without the patch step, the original images would be resized and lose resolution.

- Step 5: Train the Teacher model with PatchTeacher_Training.py
    - This script trains a teacher model. The model selected is RFDETR.
    - The transformer based RFDETR has been shown to be more performative at lower latencies for BoundingBox classification

- Step 6 \[Optional]: Evaluate the performance of the trained model visually with Patch_Inference_RFDETR_Testing.py
    - This script loads labeled images in "real-time", patches the image, performs inference on each patch (batched), and then applies the bbox inferences back to the original images to visualize
    - The script can be used to evaluate if the teacher model + patching process are correctly working in concert before moving on to batch labeling for the student

- Step 7: Label all images at scale with the teacher model using Create_Student_Dataset.py
    - This script uses the trained Teacher model to perform batch-inference across the whole dataset
    - The predictions are coalated for each image, so the student will be able to train on images as a whole
        - The YOLOv8n studen is Conv based, and should be more performative on larger high-resolution images


## Student \[WIP]
- Step 8: Train student
    - This is currently a work in progress


## Agent \[WIP]
- step N: TBD