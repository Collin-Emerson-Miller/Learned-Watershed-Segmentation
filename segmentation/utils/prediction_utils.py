
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models import BachNet
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from . import preprocessing_utils


# In[2]:


def input_generator(bach, train_path, input_path, tag):

    image_data = dict()

    files = os.listdir(os.path.join(train_path, input_path))

    for i, filename in enumerate(files):
        f_name, ext = os.path.splitext(filename)
        print("Loading image: ", f_name)


        if "gt" in f_name.split("_"):
            continue

        gt_filename = f_name + "_" + tag

        gt_path = os.path.join(train_path, input_path, (gt_filename + ext))
        if not os.path.isfile(gt_path):
            continue    
            
        print("Loading image: ", f_name)

        foldername = "data/train/chopin/" + f_name

        image_path = os.path.join(train_path, input_path, filename)

        img, gt, gt_cuts, seeds = preprocessing_utils.load_image(foldername,
                                        image_path,
                                        gt_path) 
        
        seed_image = gt

        for x, y in seeds:
            seed_image[x, y] = [255, 0, 0]

        plt.imsave(os.path.join(foldername, "seed_image.png"), seed_image, cmap='gray')

        bps = bach.boundary_probabilities(img)
        
        I_a = np.stack((img, bps), axis=-1)

        plt.imsave(os.path.join(train_path, "bach", f_name + ".png"), bps, cmap='gray')
            
        yield f_name, img, bps, I_a, gt, gt_cuts, seeds
    
    
