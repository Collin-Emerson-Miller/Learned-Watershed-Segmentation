#! /usr/bin/env python

from __future__ import division
from __future__ import print_function

import h5py
import numpy as np
import sys


def ids_to_segmentation(neuron_ids):
    """
    Given a numpy array containing the neuron ids for a CREMI neural segmentation image,
    this function returns the ground segmentation image where a segmentation boundary
    is a white pixel and everything else is a black pixel.

    Args:
        neuron_ids: A numpy array containing the neuron ids of a segmentation image.

    Returns:
        A numpy array of ground truth segmentations.

    """

    edges_right = neuron_ids[:, :-1].ravel() != neuron_ids[:, 1:].ravel()
    edges_down = neuron_ids[:-1].ravel() != neuron_ids[1:].ravel()

    edges_down = edges_down.reshape((neuron_ids.shape[0] - 1, neuron_ids.shape[1]))
    edges_right = edges_right.reshape((neuron_ids.shape[0], neuron_ids.shape[1] - 1))

    segmentations = np.zeros_like(neuron_ids)

    for index, x in np.ndenumerate(edges_down):
        if x:
            segmentations[index] = 255
            segmentations[index[0] + 1][index[1]] = 255

    for index, x in np.ndenumerate(edges_right):
        if x:
            segmentations[index] = 255
            segmentations[index[0]][index[1] + 1] = 255

    return segmentations

# Read file.
hfile = sys.argv[1]

# Get neuron ids from Cremi data.
with h5py.File(hfile, "r") as hdf:
    print ("Initializing...")
    labels = hdf['volumes']['labels']['neuron_ids'][:]
    print ("Done!")

# Initialize segmentation image placeholder.
gt_segmentations = np.zeros_like(labels)

# Iterate through slices and save segmentations to
# ground truth segmentations array.
print("Converting to ground truth segmentations.")
print("This may take a while...")
for index, x in enumerate(labels):
    progress = index/len(labels) * 100
    sys.stdout.write("\rProgress\t[%d%%]" % progress)
    sys.stdout.flush()
    gt_segmentations[index] = ids_to_segmentation(x)

# Write segmentations to disk and close file.
filename = hfile.split('.')
filename.insert(1, '_segmentation.')
filename = ''.join(filename)

hdf_file = h5py.File(filename, 'w')
hdf_file.create_dataset(hfile, data=gt_segmentations)
hdf_file.close()

print("Conversion Successful.")