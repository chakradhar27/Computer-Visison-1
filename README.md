# Industrial-Computer-Vision-Project
## Non-invasive detection, tracking and classification of insects.

The aim of the project was to identify the key problems in the field of image analysis to track very small objects like insects, look for any alternative approaches in terms of finding promising solutions that are being applied and different data sources and to apply deep learning (Computer Vision/ML) techniques for the detection and classification of insects and insects behaviours. 
The objective of this study was to analyse the problem of pixel-ratio, the proportion of the target object (insect) in the image by applying the low computing object detection algorithms such as SSD Mobilenet, on the large insect datasets that are collected. A customised Cross-Validation model architecture was built completely in Python (Tensorflow) by training multiple variants of SSD Mobilenet V2 on a set of images (with certain proportion of target object) and validated the models performance in terms of detection and classification on different sets of scaled images (with very low and very high proportion of target object) to understand the generalizability and adaptability of the Mobilenet models in the case of Non-invasive approach.

## Instructions

1. The insect data used for teh study can be found in insect.rar.
2. The data is annotated already using the image annotation software and the xml files are also present within the rar file.
3. To run the models you can use "Custom_OD_insekten_SSD_MB_640.py" and "Custom_OD_kaggle_SSD_MB_320.py" files and execute the code in a notebook.
4. The saved models and checkpoints are captured and stored in the "ssd_mb_640" and "ssd_mb_320" folders.
