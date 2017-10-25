from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import gc
import keras
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
import sys

from models import ChopinNet
from models import BachNet
from utils import display_utils
from utils import graph_utils
from utils import prediction_utils
from utils import preprocessing_utils

def train_single(chopin,
                 img,
                 I_a,
                 gt,
                 seeds,
                 foldername,
                 global_loss=[],
                 global_accuracy=[],
                 num_epochs=8):
    I_a = preprocessing_utils.pad_for_window(I_a,
                                             chopin.receptive_field_shape[0],
                                             chopin.receptive_field_shape[1])

    graph = graph_utils.prims_initialize(img)

    ground_truth_cuts, gt_assignments = graph_utils.generate_gt_cuts(gt,
                                                                     seeds,
                                                                     assignments=True)

    local_loss = []
    local_accuracy = []

    os.mkdir(os.path.join(foldername, "saved_models"))

    for epoch in range(num_epochs):
        print("Epoch {}".format(epoch + 1))
        msf = prediction_utils.minimum_spanning_forest(chopin, I_a, graph, seeds)

        segmentations = display_utils.assignments(img, msf, seeds)

        shortest_paths = nx.get_node_attributes(msf, 'path')
        assignments = nx.get_node_attributes(msf, 'seed')
        cuts = graph_utils.get_cut_edges(msf)

        acc = graph_utils.accuracy(assignments, gt_assignments)

        print("Accuracy: ", acc)
        local_accuracy.append(acc)

        filename = "epoch_{}.png".format(epoch + 1)

        boundaries = display_utils.view_boundaries(np.zeros_like(img),
                                                   cuts)

        plt.imsave(os.path.join(foldername, filename), boundaries)

        constrained_msf = msf.copy()

        constrained_msf.remove_edges_from(ground_truth_cuts)

        constrained_msf = graph_utils.minimum_spanning_forest(constrained_msf,
                                                              seeds)

        ground_truth_paths = nx.get_node_attributes(constrained_msf, 'path')

        children = graph_utils.compute_root_error_edge_children(shortest_paths,
                                                                ground_truth_paths,
                                                                cuts,
                                                                ground_truth_cuts)

        weights = []
        static_images = []
        dynamic_images = []

        loss = 0

        for (u, v), weight in children.iteritems():
            weights.append(weight)
            static_images.append(msf.get_edge_data(u, v)['static_image'][0])
            dynamic_images.append(msf.get_edge_data(u, v)['dynamic_image'][0])
            altitude_val = msf.get_edge_data(u, v)['weight']

            loss += weight * altitude_val

        batches = zip(preprocessing_utils.create_batches(np.expand_dims(np.stack(weights), 1)),
                      preprocessing_utils.create_batches(np.stack(static_images)),
                      preprocessing_utils.create_batches(np.stack(dynamic_images)))

        with chopin.sess.as_default():
            chopin.sess.run(chopin.zero_ops)

            for w, s, d in batches:
                feed_dict = {chopin.gradient_weights: w.transpose(),
                             chopin.static_input: s,
                             chopin.dynamic_input: d,
                             keras.backend.learning_phase(): 0}

                chopin.sess.run(chopin.accum_ops, feed_dict)

            chopin.sess.run(chopin.train_step)

        local_loss.append(loss)
        print("Loss: ", loss)

        info = "Epoch: {}\tloss: {}\taccuracy: {}\n".format(epoch + 1, loss, acc)
        loss_file.write(info)
        loss_file.flush()

        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(local_loss)
        axarr[0].set_title("Loss")
        axarr[1].plot(local_accuracy)
        axarr[1].set_title("Accuracy")

        figurename = "Local Loss and Accuracy"

        plt.savefig(os.path.join(foldername, figurename))

        global_loss.append(loss)
        global_accuracy.append(acc)
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(global_loss)
        axarr[0].set_title("Loss")
        axarr[1].plot(global_accuracy)
        axarr[1].set_title("Accuracy")

        figurename = "Global Loss and Accuracy"

        global_folder = foldername.split("/")[:-1]
        global_folder = "/".join(global_folder)

        plt.savefig(os.path.join(global_folder, figurename))

        chopin.save_model("models/saved_model/Chopin/model.ckpt")
        model_name = "epoch_{}".format(epoch)
        chopin.save_model(os.path.join(foldername, "saved_models", model_name, model_name))
    gc.collect()

    return segmentations, global_loss, global_accuracy

receptive_field_shape = (23, 23)
bach = BachNet.BachNet()

bach.build(23, 23, 1)
bach.load_model('models/saved_model/Bach/model.h5')

images_to_train = file(sys.argv[1], "r")

image_data = dict()

for line in images_to_train:
    f_name, image_path, gt_path = line.split()

    sys.stdout.write("\rLoading Image: {}".format(f_name))

    foldername = os.path.join("data/train/chopin", f_name)

    img, gt, seeds = preprocessing_utils.load_image(foldername,
                                                    f_name,
                                                    image_path,
                                                    gt_path)

    bps = prediction_utils.boundary_probabilities(bach, img)

    image_data[f_name] = img, bps, gt, seeds

chopin = ChopinNet.Chopin()
chopin.build(receptive_field_shape)
chopin.initialize_session()
chopin.load_model("models/saved_model/Chopin/model.ckpt")

global_loss = list()
global_accuracy = list()

try:
    loss_file
except NameError:
    loss_file = open('data/train/chopin/global_loss.txt', 'w')

for name, (img, bps, gt, seeds) in image_data.iteritems():
    print("Training on " + name)

    img_info = "\nImage: {}\n\n".format(name)
    loss_file.write(img_info)
    loss_file.flush()

    I_a = np.stack((img, bps), axis=2)
    foldername = "data/train/chopin/" + name
    segs, glob, acc_timeline = train_single(chopin,
                                            img,
                                            I_a,
                                            gt,
                                            seeds,
                                            foldername,
                                            global_loss,
                                            global_accuracy,
                                            num_epochs=16)

loss_file.close()


