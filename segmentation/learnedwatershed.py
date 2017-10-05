# Crafted by Collin Miller

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from models import ChopinNet
from utils import preprocessing_utils
from utils import prediction_utils

def segment(img, presegmentations):
    """Segments an image with learned watershed technique.

    Args:
        img: A `numpy` array that is the input image.
        presegmentations: A `numpy` array of  presegmentations to generate seeds from.

    Returns:
        A `numpy` array of segmentation ids.
    """

    '''
    A temporary folder is created so that the seed generation command can be run on an image.
    Therefore, if that temp folder already exists, then it must be deleted.
    '''
    if os.path.exists('temp'):
        shutil.rmtree('temp')

    os.makedirs('temp')

    '''
    Now that the temp folder has been made, the seeds can be generated on the image.
    '''

    plt.imsave("temp/presegmentaitons.png", presegmentations)
    seeds = preprocessing_utils.generate_seeds(presegmentations, 'temp/')

    shutil.rmtree('temp')

    '''
    After the seeds are generated, the segmentation process can start.
    '''

    '''
    Now that the segmentation is complete, the segment ids can be returned.
    '''
    return("Done!")


if os.path.exists('temp'):
    shutil.rmtree('temp')

os.makedirs('temp')
img = cv2.imread("data/train/input/slice_0.png", 0)
presegmentation = cv2.imread("data/train/input/slice_0_gt.png", 0)

plt.imsave("a", presegmentation)

print(presegmentation.shape)
segment(img, "data/train/input/slice_0_gt.png")



