from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import re
import shutil

import numpy as np
import math
import matplotlib.pyplot as plt


def load_image(foldername, img_path, gt_path):
    
    if os.path.exists(foldername):
        shutil.rmtree(foldername)

    os.makedirs(foldername)
    
    img = cv2.imread(img_path, 0)
    gt = cv2.imread(gt_path, 0)
    
    plt.imsave(os.path.join(foldername, "img"), img, cmap='gray')
    plt.imsave(os.path.join(foldername, "gt"), gt, cmap='gray')    

    seeds = generate_seeds(gt_path, foldername)
    
    labels_path = os.path.join(foldername, "labels.png")
    
    labels = cv2.imread(labels_path, 0)
    
    gt_cuts = get_gt_cuts(labels)
    
    return img, gt, gt_cuts, seeds


def generate_seeds(image_path, output_path):
    """Generates a list of seeds for the image at a specified path.

    Args:
        image_path: The path to the image in a folder.
        output_path: The path to save the list of seeds.

    Returns:
        A list of seeds.
    """
    
    seed_path = os.path.join(output_path, "seeds.txt")
    labels_path = os.path.join(output_path, "labels.png")
    
    os.system("gmic -v -1 " + image_path + " -negate -label_fg 0,0 -dilate_circ 6 -o " + labels_path + " -o -.asc | tail -n +2 | awk '{ for (i = 1; i<=NF; i++) {x[$i] += i; y[$i] += NR; n[$i]++; } } END { for (v in x) { if (v>0) print v,x[v]/n[v],y[v]/n[v] }}' > " + seed_path + "")

    seeds = []
    f = open(output_path + "/seeds.txt", 'r')
    for line in f:
        y = int(float(re.split(' ', line)[1]))
        x = int(float(re.split(' ', line)[2]))
        seed = (x - 1, y - 1)
        seeds.append(seed)

    return seeds[:]


def get_gt_cuts(labels):
    edges_right = labels[:, :-1].ravel() != labels[:, 1:].ravel()
    edges_down = labels[:-1].ravel() != labels[1:].ravel()

    edges_down = edges_down.reshape((labels.shape[0] - 1, labels.shape[1]))
    edges_right = edges_right.reshape((labels.shape[0], labels.shape[1] - 1))

    ground_truth_cuts = []

    for index, x in np.ndenumerate(edges_down):
        if x:
            ground_truth_cuts.append(((index), (index[0] + 1, index[1])))

    for index, x in np.ndenumerate(edges_right):
        if x:
            ground_truth_cuts.append(((index), (index[0], index[1] + 1)))
            
    return ground_truth_cuts


def crop_2d(image, top_left_corner, height, width):
    """
    Returns a crop of an image.

    Args:
        image: The original image to be cropped.
        top_left_corner: The coordinates of the top left corner of the image.
        height: The hight of the crop.
        width: The width of the crop.

    Returns:
        A cropped version of the original image.
    """

    x_start = top_left_corner[0]
    y_start = top_left_corner[1]
    x_end = x_start + width
    y_end = y_start + height

    return image[x_start:x_end, y_start:y_end, ...]


# In[ ]:


def pad_for_window(img, height, width, padding_type='reflect'):
    npad = ((height // 2, width // 2), (height // 2, width // 2), (0, 0))
    return np.pad(img, npad, padding_type)


# In[ ]:


def prepare_input_images(img, height=23, width=23):
    """
    Preprocess images to be used in the prediction of the edges.

    Args:
        image (numpy.array):
    """

    # Standardize input
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)

    padded_image = pad_for_window(img, height, width)

    images = []

    for index in np.ndindex(img.shape[:-1]):
        images.append(crop_2d(padded_image, index, height, width))

    return np.stack(images)


# In[ ]:


def create_batches(x, max_batch_size=32):
    """

    Args:
        x: A numpy array of the input data
        y: A numpy array of the output
        max_batch_size: The maximum elements in each batch.

    Returns: A list of batches.

    """

    batches = math.ceil(x.shape[0] / max_batch_size)
    x = np.array_split(x, batches)

    return x

