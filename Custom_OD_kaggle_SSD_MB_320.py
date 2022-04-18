#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib

# Clone the tensorflow models repository if it doesn't already exist
if "models" in pathlib.Path.cwd().parts:
  while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')
elif not pathlib.Path('models').exists():
  get_ipython().system('git clone --depth 1 https://github.com/tensorflow/models')


# In[2]:


# Install the Object Detection API
%%bash
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .


# In[3]:


get_ipython().system('unzip Custom_OD')


# In[5]:


#Test if there any corrupted images
TRAIN_IMAGE_FILE = '/Custom_OD/Workspace/images/Train'
import glob
from skimage import io
import cv2
files=[]
files=glob.glob(TRAIN_IMAGE_FILE+ '\*.jpg')
for i in range(len(files)):
        
    try:
        _ = io.imread(files[i])
        img = cv2.imread(files[i])
   
    except Exception as e:
        print(e)
        print(files[i])


# In[6]:


get_ipython().run_cell_magic('bash', '', '\n# Create train data:\npython generate_tfrecord.py -x /content/Custom_OD/Workspace/images/Train -l /content/Custom_OD/Workspace/Annotations/label_map.pbtxt -o /content/Custom_OD/Workspace/Annotations/train.record\n\n# Create test data:\npython generate_tfrecord.py -x /content/Custom_OD/Workspace/images/test -l /content/Custom_OD/Workspace/Annotations/label_map.pbtxt -o /content/Custom_OD/Workspace/Annotations/test.record')


# In[7]:


# Load the TensorBoard notebook extension
get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[8]:


get_ipython().run_line_magic('tensorboard', '--logdir /content/Custom_OD/Workspace/models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8')


# In[9]:


get_ipython().run_cell_magic('bash', '', 'cd Custom_OD/Workspace/\npython model_main_tf2.py --model_dir=/content/Custom_OD/Workspace/models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8 --pipeline_config_path=/content/Custom_OD/Workspace/models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config')


# In[10]:


get_ipython().run_cell_magic('bash', '', 'cd Custom_OD/Workspace/\npython exporter_main_v2.py --input_type=image_tensor --pipeline_config_path=/content/Custom_OD/Workspace/models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config --trained_checkpoint_dir=/content/Custom_OD/Workspace/models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8 --output_directory=/content/Custom_OD/Workspace/exported_model/ssd_mb_320')


# In[28]:


#Import the required libraries for Object detection infernece
import time
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')
# setting min confidence threshold
MIN_CONF_THRESH=.6
#Loading the exported model from saved_model directory
PATH_TO_SAVED_MODEL =r'/content/Custom_OD/Workspace/exported_model/ssd_mb_320/saved_model'
print('Loading model...', end='')
start_time = time.time()
# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))
# LOAD LABEL MAP DATA
PATH_TO_LABELS=r'/content/Custom_OD/Workspace/Annotations/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,                                                                    use_display_name=True)

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array of shape (height, width, channels), where channels=3 for RGB to feed into tensorflow graph.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(image_path))

def show_inference(detect_fn, Image_path):
    image_np = load_image_into_numpy_array(Image_path)
    # Running the infernce on the image specified in the  image path
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
               for key, value in detections.items()}
    detections['num_detections'] = num_detections
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    #print(detections['detection_classes'])
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=MIN_CONF_THRESH,
      agnostic_mode=False)
    
    display(Image.fromarray(image_np_with_detections))


# In[29]:


#Image files for inference
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('/content/Custom_OD/Workspace/images/Test')
TEST_IMAGEs_PATH = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
for image_path in TEST_IMAGEs_PATH:
  print(image_path)
  show_inference(detect_fn, image_path)

