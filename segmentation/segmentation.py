from __future__ import print_function

import BachNet
import ChopinNet
import utils
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model


def watershed(img, seeds):
    bach = BachNet.BachNet()

    boundary_probabilities = bach.boundary_probabilities(img, verbose=1)

    I_a = np.stack((img, boundary_probabilities), axis=2)

    images_in = utils.prepare_input_images(I_a)

    chopin = ChopinNet.ChopinNet()

    altitudes = chopin.predict_altitudes(images_in, 1000)
    altitudes = np.reshape(altitudes, img.shape)

    # Compute the cut edges and the shortest paths.
    graph = utils.prims_initialize(img)

    for (x, y), d in np.ndenumerate(altitudes):
        graph.node[(x, y)]['altitude'] = d

    graph = utils.minimum_spanning_forest(graph, seeds)

    return utils.assignments(img, graph, seeds)
