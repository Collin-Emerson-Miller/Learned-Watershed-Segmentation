from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import matplotlib.pyplot as plt
import numpy as np

def view_path(image, path):

    img = image.copy()

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for x, y in path:
        img[x, y] = [0, 0, 255]
    return img


def view_boundaries(image, cuts):

    img = image.copy()

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for x, y in cuts:
        img[x[0], x[1]] = [255, 255, 255]
        img[y[0], y[1]] = [255, 255, 255]

    return img


def get_spaced_colors(n):
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


def transparent_mask(img, segmentations, alpha=0.5):

    output = img.copy()
    if len(output) <= 2:
    	output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

    segmentations = segmentations.astype('uint8')
    overlay = segmentations
    output = cv2.addWeighted(overlay, alpha, output, 1 - alpha,
                0, output)

    return output


def assignments(img, graph, seeds):
    assignment_mask = np.zeros((img.shape[0], img.shape[1], 3))

    colors = get_spaced_colors(len(seeds) + 1)

    for node, d in graph.nodes_iter(data=True):
        seed = d['seed']
        try:
            assignment_mask[node] = colors[seeds.index(seed) + 1]
        except ValueError:
            assignment_mask[node] = colors[0]
            
    return assignment_mask

