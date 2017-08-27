from __future__ import print_function

import keras
import time
import numpy as np
import tensorflow as tf
import utils
import BachNet
import math
import networkx as nx


class ChopinNet:

    def __init__(self):
        pass

    def _build(self, width, height, channels):
        """
        Builds

        Args:
            i_augmented: rgb image [batch, height, width, 2]
        """

        start_time = time.time()
        #print("\nBuild model started.")


        self.i_augmented = tf.placeholder(tf.float32,
                                           shape=(None, width, height, channels))

        m = keras.layers.Conv2D(16, 5, padding='same',
                                activation='elu', dilation_rate=1)(self.i_augmented)
        m = keras.layers.Conv2D(16, 3, padding='same',
                                activation='elu', dilation_rate=1)(m)
        m = keras.layers.BatchNormalization()(m)
        m = keras.layers.Conv2D(32, 3, padding='same',
                                activation='elu', dilation_rate=2)(m)
        m = keras.layers.BatchNormalization()(m)
        m = keras.layers.Conv2D(32, 3, padding='same',
                                activation='elu', dilation_rate=4)(m)
        m = keras.layers.BatchNormalization()(m)
        m = keras.layers.Conv2D(64, 3, padding='same',
                                activation='elu', dilation_rate=8)(m)
        m = keras.layers.BatchNormalization()(m)
        m = keras.layers.Conv2D(64, 3, padding='same',
                                activation='elu', dilation_rate=16)(m)
        m = keras.layers.BatchNormalization()(m)
        m = keras.layers.Conv2D(128, 3, padding='same',
                                activation='elu', dilation_rate=1)(m)
        m = keras.layers.BatchNormalization()(m)
        m = keras.layers.Flatten()(m)
        m = keras.layers.Dense(16, activation='relu')(m)
        m = keras.layers.BatchNormalization()(m)
        m = keras.layers.Dense(1, activation='elu')(m)
        self.f_static = keras.layers.BatchNormalization()(m)


        # This placeholder will hold the root error edge values.
        self.gradient_weights = tf.placeholder(tf.float32, shape=(1, None))

        # Define optimizer
        opt = tf.train.GradientDescentOptimizer(learning_rate=0.000001)

        tvs = tf.trainable_variables()

        loss = tf.matmul(self.gradient_weights, self.f_static)

        # Accumulate gradients of predictions with respect to the parameters.
        accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

        gvs = opt.compute_gradients(loss, tvs)
        self.accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]

        # Apply gradients
        self.train_step = opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])

        #print(("Build model finished: %fs" % (time.time() - start_time)))

    def predict_altitudes(self, images, max_batch_size):
        """

        Args:
            images:

        Returns:

        """

        tf.reset_default_graph()
        keras.backend.clear_session()
        self._build(images.shape[1], images.shape[2], images.shape[3])

        saver = tf.train.Saver()

        with tf.Session() as sess:

            try:
                # Restore variables from disk.
                saver.restore(sess, "saved_model/Chopin/model.ckpt")
                #print("Model restored.")
            except:
                sess.run(tf.global_variables_initializer())
                save_path = saver.save(sess, "saved_model/Chopin/model.ckpt")
                #print("Model saved in file: %s" % save_path)


            num_batches = math.ceil(images.shape[0] / max_batch_size)
            batches = np.array_split(images, num_batches)

            altitudes = []

            times = []

            print("Starting predictions")

            for batch in batches:
                feed_dict = {self.i_augmented: batch,
                             keras.backend.learning_phase(): 0}

                start = time.time()
                altitudes.append(sess.run(self.f_static, feed_dict=feed_dict))

                end = time.time()
                times.append(end - start)
            print("Prediction Done.  Average time: %fs, Total time: %fs" % (np.mean(times), np.sum(times)))
            altitudes = np.vstack(altitudes)

        return altitudes


def _training_epoch(img, images_in, gt, seeds):
    graph = utils.prims_initialize(img)

    chopin = ChopinNet()

    altitudes = chopin.predict_altitudes(images_in, 1000)
    altitudes = np.reshape(altitudes, img.shape)

    for (x, y), d in np.ndenumerate(altitudes):
        graph.node[(x, y)]['altitude'] = d
        image = images_in[x + y]
        graph.node[(x, y)]['image'] = image

    print("Generating MSF")
    msf = utils.minimum_spanning_forest(graph, seeds)

    segmentations = utils.assignments(img, msf, seeds)

    shortest_paths = nx.get_node_attributes(msf, 'path')
    assignments = nx.get_node_attributes(msf, 'seed')
    cuts = utils.get_cut_edges(msf)

    print("Generating Ground Truth Graph")
    ground_truth_cuts, gt_assignments = utils.generate_gt_cuts(gt, seeds, assignments=True)

    acc = utils.accuracy(assignments, gt_assignments)

    constrained_graph = graph.copy()

    constrained_graph.remove_edges_from(ground_truth_cuts)

    print("Generating Constrained MSF")
    constrained_msf = utils.minimum_spanning_forest(constrained_graph, seeds)

    ground_truth_paths = nx.get_node_attributes(constrained_msf, 'path')

    children = utils.compute_root_error_edge_children(shortest_paths,
                                                      ground_truth_paths, cuts,
                                                      ground_truth_cuts)

    images = []
    weights = []
    loss = 0

    for edge, weight in children.iteritems():
        weights.append(weight)
        image = graph.edge[edge[0]][edge[1]]['image']
        images.append(image)
        altitude = graph.edge[edge[0]][edge[1]]['weight']

        loss += altitude * weight

    batches = utils.create_batches(np.array(images), np.array(weights))

    saver = tf.train.Saver()

    with tf.Session() as sess:

        try:
            # Restore variables from disk.
            saver.restore(sess, "saved_model/Chopin/model.ckpt")
            #print("Model restored.")
        except:
            sess.run(tf.global_variables_initializer())
            save_path = saver.save(sess, "saved_model/Chopin/model.ckpt")
            #print("Model saved in file: %s" % save_path)

        # Zero out gradient accumulator.
        sess.run(chopin.zero_ops)

        # Accumulate gradients.
        for batch in batches:
            sess.run(chopin.accum_ops, feed_dict={chopin.i_augmented: batch[0],
                                                  chopin.gradient_weights: [batch[1]],
                                                  keras.backend.learning_phase(): 0})

        sess.run(chopin.train_step)

        save_path = saver.save(sess, "saved_model/Chopin/model.ckpt")
        #print("Model saved in file: %s" % save_path)

    return segmentations, loss, acc


def fit(img, gt, seeds, num_epochs):
    bach = BachNet.BachNet()

    boundary_probabilities = bach.boundary_probabilities(img, verbose=1)

    I_a = np.stack((img, boundary_probabilities), axis=2)

    images_in = utils.prepare_input_images(I_a)

    loss_timeline = []
    segmentation_epochs = []
    accuracy_timeline = []

    for epoch in range(num_epochs):

        print("\nEpoch {}".format(epoch + 1))
        segmentations, loss, acc = _training_epoch(img, images_in, gt, seeds)
        loss_timeline.append(loss)
        segmentation_epochs.append(segmentations)
        accuracy_timeline.append(acc)

    return boundary_probabilities, segmentation_epochs, loss_timeline, accuracy_timeline


def predict(img, seeds):
    bach = BachNet.BachNet()

    boundary_probabilities = bach.boundary_probabilities(img, verbose=1)

    I_a = np.stack((img, boundary_probabilities), axis=2)

    images_in = utils.prepare_input_images(I_a)

    graph = utils.prims_initialize(img)

    chopin = ChopinNet()

    altitudes = chopin.predict_altitudes(images_in, 1000)
    altitudes = np.reshape(altitudes, img.shape)

    for (x, y), d in np.ndenumerate(altitudes):
        graph.node[(x, y)]['altitude'] = d
        image = images_in[x + y]
        graph.node[(x, y)]['image'] = image

    print("Generating MSF")
    msf = utils.minimum_spanning_forest(graph, seeds)

    segmentations = utils.assignments(img, msf, seeds)

    return segmentations