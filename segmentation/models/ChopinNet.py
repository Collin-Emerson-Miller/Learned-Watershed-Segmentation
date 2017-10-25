from __future__ import print_function

import keras
import time
import numpy as np
import tensorflow as tf
import utils
import BachNet
import os
import networkx as nx
import matplotlib.pyplot as plt
import gc
from heapq import heappush as push
from heapq import heappop as pop
from utils import display_utils
from utils import graph_utils
from utils import preprocessing_utils
from utils import relative_assignments


class Chopin:
    def __init__(self):
        self.sess = tf.Session()

    def build(self, receptive_field_shape, learning_rate=0.001):

        self.receptive_field_shape = receptive_field_shape

        tf.reset_default_graph()
        keras.backend.clear_session()

        self.static_input = tf.placeholder(tf.float32,
                                           shape=(None, self.receptive_field_shape[0],
                                                  self.receptive_field_shape[1],
                                                  2))

        self.dynamic_input = tf.placeholder(tf.float32,
                                            shape=(None, self.receptive_field_shape[0],
                                                   self.receptive_field_shape[1], 3))

        # Static Body.
        static = keras.layers.Conv2D(16, 5, padding='same',
                                     activation='elu', dilation_rate=1)(self.static_input)
        static = keras.layers.BatchNormalization()(static)
        static = keras.layers.Conv2D(16, 3, padding='same',
                                     activation='elu', dilation_rate=1)(static)
        static = keras.layers.BatchNormalization()(static)
        static = keras.layers.Conv2D(32, 3, padding='same',
                                     activation='elu', dilation_rate=2)(static)
        static = keras.layers.BatchNormalization()(static)
        static = keras.layers.Conv2D(32, 3, padding='same',
                                     activation='elu', dilation_rate=4)(static)
        static = keras.layers.BatchNormalization()(static)
        static = keras.layers.Conv2D(64, 3, padding='same',
                                     activation='elu', dilation_rate=8)(static)
        static = keras.layers.BatchNormalization()(static)
        static = keras.layers.Conv2D(64, 3, padding='same',
                                     activation='elu', dilation_rate=16)(static)
        static = keras.layers.BatchNormalization()(static)
        static = keras.layers.Conv2D(128, 3, padding='same',
                                     activation='elu', dilation_rate=1)(static)
        static = keras.layers.BatchNormalization()(static)
        static = keras.layers.Flatten()(static)

        # Dynamic body.
        dynamic = keras.layers.Conv2D(32, 3, padding='same',
                                      activation='elu', dilation_rate=4)(self.dynamic_input)
        dynamic = keras.layers.BatchNormalization()(dynamic)
        dynamic = keras.layers.Conv2D(32, 3, padding='same',
                                      activation='elu', dilation_rate=8)(dynamic)
        dynamic = keras.layers.BatchNormalization()(dynamic)
        dynamic = keras.layers.Conv2D(64, 3, padding='same',
                                      activation='elu', dilation_rate=16)(dynamic)
        dynamic = keras.layers.BatchNormalization()(dynamic)
        dynamic = keras.layers.Conv2D(64, 3, padding='same',
                                      activation='elu', dilation_rate=1)(dynamic)
        dynamic = keras.layers.BatchNormalization()(dynamic)
        dynamic = keras.layers.Flatten()(dynamic)

        merge = keras.layers.concatenate([static, dynamic], 1)

        self.altitude = keras.layers.Dense(1024, activation='relu')(merge)
        self.altitude = keras.layers.BatchNormalization()(self.altitude)
        self.altitude = keras.layers.Dense(1, activation='elu')(self.altitude)

        # This placeholder will hold the root error edge values.
        self.gradient_weights = tf.placeholder(tf.float32, shape=(1, None))

        # Define optimizer
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        tvs = tf.trainable_variables()

        self.loss = tf.matmul(self.gradient_weights, self.altitude)

        # Accumulate gradients of predictions with respect to the parameters.
        accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

        gvs = opt.compute_gradients(self.loss, tvs)
        self.accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]

        # Apply gradients
        self.train_step = opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])

    def predict_altitudes(self, static_images, dynamic_images):
        """
        Predicts the altitude of one or more edges given the static image and the dynamic image.

        Args:

            static_images: a numpy.ndarray of shape [None, receptive_field_shape[0],
            receptive_field_shape[1], 2] images from the original image that are
            augmented with boundary probabilities and are cropped to the same size
            of the receptive field.

            dynamic_images: a numpy.ndarray of rgb images of shape [None, receptive_field_shape[0],
            receptive_field_shape[1], 3] that represent the relative assignments.

        Returns:
            The altitudes of the edges.
        """

        with self.sess.as_default():
            feed_dict = {self.static_input: static_images,
                         self.dynamic_input: dynamic_images,
                         keras.backend.learning_phase(): 0}

            altitudes = self.sess.run(self.altitude, feed_dict)

        return altitudes

    def initialize_session(self):
        """
        Initializes a session in TensorFlow.
        """

        if not self.sess._closed:
            self.sess.close()

        self.sess = tf.InteractiveSession()

        self.sess.run(tf.global_variables_initializer())

    def load_model(self, filepath):
        saver = tf.train.Saver()

        with self.sess.as_default():
            try:
                # Restore variables from disk.
                saver.restore(self.sess, filepath)
                # print("Model restored.")
            except:
                self.sess.run(tf.global_variables_initializer())
                save_path = saver.save(self.sess, filepath)

    def save_model(self, filepath):
        saver = tf.train.Saver()

        with self.sess.as_default():
            save_path = saver.save(self.sess, filepath)

    def predicted_msf(self, I_a, graph, seeds):
        num_nodes = graph.number_of_nodes()
        visited = np.zeros(I_a.shape[:-1])
        frontier = []

        ra = relative_assignments.RelativeAssignments(seeds,
                                                      (I_a.shape[0],
                                                       I_a.shape[1]),
                                                      self.receptive_field_shape)
        static_input_images = preprocessing_utils.prepare_input_images(I_a, height=self.receptive_field_shape[0],
                                                                       width=self.receptive_field_shape[1])

        print("Starting gradient segmentation...")
        start = time.time()

        for u in seeds:

            # Assign seed to chopin.
            graph.node[u]['seed'] = u

            ra.assign_node(u, seeds.index(u))

            visited[u[0], u[1]] = 1

            # Store path.
            graph.node[u]['path'] = [u]

            # Push all edges
            static_input = []
            dynamic_input = []
            edges = []
            for u, v in graph.edges(u):
                edges.append((u, v))
                seed_index = seeds.index(graph.node[u]['seed'])
                static_image = static_input_images[v[0] * I_a.shape[1] + v[1]]
                dynamic_image = ra.prepare_images([(u, seed_index)])[0]
                static_input.append(static_image)
                dynamic_input.append(dynamic_image)

                try:
                    graph.edge[u][v]['static_image'] = static_image
                    graph.edge[u][v]['dynamic_image'] = dynamic_image
                except KeyError:
                    pass

            static_input = np.stack(static_input)
            dynamic_input = np.stack(dynamic_input)

            altitude_values = self.predict_altitudes(static_input,
                                                      dynamic_input)

            for (u, v), alt in zip(edges, altitude_values):
                graph.edge[u][v]['weight'] = alt[0]
                push(frontier, (alt, u, v))

        while frontier:
            W, u, v = pop(frontier)


            if visited[v[0], v[1]] == 1:
                continue

            # Assign the node
            graph.node[v]['seed'] = graph.node[u]['seed']

            ra.assign_node(v, seeds.index(graph.node[u]['seed']))

            # Store path.
            graph.node[v]['path'] = graph.node[u]['path'] + [v]

            visited[v[0], v[1]] = 1


            static_input = []
            dynamic_input = []
            edges = []
            for v, w in graph.edges(v):
                if visited[w[0], w[1]] == 0:
                    edges.append((v, w))
                    seed_index = seeds.index(graph.node[v]['seed'])
                    static_image = static_input_images[w[0] * I_a.shape[1] + w[1]]
                    dynamic_image = ra.prepare_images([(v, seed_index)])[0]
                    static_input.append(static_image)
                    dynamic_input.append(dynamic_image)

                    try:
                        graph.edge[u][v]['static_image'] = static_image
                        graph.edge[u][v]['dynamic_image'] = dynamic_image
                    except KeyError:
                        pass

            try:
                static_input = np.stack(static_input)
                dynamic_input = np.stack(dynamic_input)

                altitude_values = self.predict_altitudes(static_input,
                                                          dynamic_input)

                for (v, w), alt in zip(edges, altitude_values):
                    graph.edge[v][w]['weight'] = alt[0]
                    push(frontier, (alt, v, w))
            except ValueError:
                pass

        end = time.time()
        print("Segmentation done: %fs" % (end - start))

        return graph
    
    def train_on_image(self, img, bps, I_a, gt, gt_cuts, seeds, graph):
        msf = self.predicted_msf(I_a, graph, seeds)
        segmentations = display_utils.assignments(np.zeros_like(img), msf, seeds)

        shortest_paths = nx.get_node_attributes(msf, 'path')
        assignments = nx.get_node_attributes(msf, 'seed')
        cuts = graph_utils.get_cut_edges(msf)

        constrained_msf = msf.copy()

        constrained_msf.remove_edges_from(gt_cuts)

        constrained_msf = graph_utils.minimum_spanning_forest(img, constrained_msf, seeds)

        ground_truth_paths = nx.get_node_attributes(constrained_msf, 'path')

        children = graph_utils.compute_root_error_edge_children(shortest_paths,
                                                          ground_truth_paths, cuts,
                                                          gt_cuts)

        weights = []
        static_images = []
        dynamic_images = []

        for (u, v), weight in children.iteritems():

            try:
                static_images.append(msf.get_edge_data(u, v)['static_image'])
                dynamic_images.append(msf.get_edge_data(u, v)['dynamic_image'])
                weights.append(weight)
                altitude_val = msf.get_edge_data(u, v)['weight']
            except KeyError:
                pass

        batches = zip(preprocessing_utils.create_batches(np.expand_dims(np.stack(weights), 1)),
                      preprocessing_utils.create_batches(np.stack(static_images)),
                      preprocessing_utils.create_batches(np.stack(dynamic_images)))


        with self.sess.as_default():
            self.sess.run(self.zero_ops)

            for w, s, d in batches:
                feed_dict = {self.gradient_weights: w.transpose(),
                             self.static_input: s,
                             self.dynamic_input: d,
                             keras.backend.learning_phase(): 0}

                self.sess.run(self.accum_ops, feed_dict)
                loss = self.sess.run(self.loss, feed_dict)
                loss = loss[0][0]

            self.sess.run(self.train_step)

        return loss, segmentations, cuts
    
    
    def fit(self, img, gt, seeds, foldername, epochs=8):
        bach = BachNet.BachNet()

        print("Starting boundary predictions")

        boundary_probabilities = bach.boundary_probabilities(img, verbose=1)

        I_a = np.stack((img, boundary_probabilities), axis=2)
        I_a = utils.pad_for_window(I_a, self.receptive_field_shape[0], self.receptive_field_shape[1])

        chopin = Chopin(self.receptive_field_shape)
        chopin.build()

        chopin.initialize_session()
        chopin.load_model("saved_model/Chopin/model.ckpt")

        graph = utils.prims_initialize(img)

        ground_truth_cuts, gt_assignments = utils.generate_gt_cuts(gt, seeds, assignments=True)

        loss_timeline = []
        acc_timeline = []

        for epoch in range(epochs):
            print("Epoch: ", epoch)

            msf = self.minimum_spanning_forest(I_a, graph, seeds)

            segmentations = utils.assignments(img, msf, seeds)

            shortest_paths = nx.get_node_attributes(msf, 'path')
            assignments = nx.get_node_attributes(msf, 'seed')
            cuts = utils.get_cut_edges(msf)

            acc = utils.accuracy(assignments, gt_assignments)
            print("Accuracy: ", acc)
            acc_timeline.append(acc)

            filename = "epoch_{}.png".format(epoch)

            mask = utils.transparent_mask(img, segmentations)

            plt.imsave(os.path.join(foldername, filename), mask)

            constrained_msf = msf.copy()

            constrained_msf.remove_edges_from(ground_truth_cuts)

            constrained_msf = self.minimum_spanning_forest(constrained_msf, seeds)

            ground_truth_paths = nx.get_node_attributes(constrained_msf, 'path')

            children = utils.compute_root_error_edge_children(shortest_paths,
                                                              ground_truth_paths, cuts,
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

            batches = zip(utils.create_batches(np.expand_dims(np.stack(weights), 1)),
                          utils.create_batches(np.stack(static_images)),
                          utils.create_batches(np.stack(dynamic_images)))

            with chopin.sess.as_default():
                chopin.sess.run(chopin.zero_ops)

                for w, s, d in batches:
                    feed_dict = {chopin.gradient_weights: w.transpose(),
                                 chopin.static_input: s,
                                 chopin.dynamic_input: d,
                                 keras.backend.learning_phase(): 0}

                    chopin.sess.run(chopin.accum_ops, feed_dict)

                chopin.sess.run(chopin.train_step)



            loss_timeline.append(loss)
            print("Loss: ", loss)

            f, axarr = plt.subplots(2, sharex=True)
            axarr[0].plot(loss_timeline)
            axarr[0].set_title("Loss")
            axarr[1].plot(acc_timeline)
            axarr[1].set_title("Accuracy")

            figurname = "loss_and_accuracy"

            plt.savefig(os.path.join(foldername, figurname))

            chopin.save_model("saved_model/Chopin/model.ckpt")

        chopin.sess.close()
        gc.collect()

        return segmentations, loss_timeline, acc_timeline

    def evaluate(self, x, y):

        bach = BachNet.BachNet()

        print("Starting boundary predictions")

        boundary_probabilities = bach.boundary_probabilities(x, verbose=1)

        I_a = np.stack((x, boundary_probabilities), axis=2)
        I_a = utils.pad_for_window(I_a, self.receptive_field_shape[0], self.receptive_field_shape[1])

        chopin = Chopin(self.receptive_field_shape)
        chopin.build()

        chopin.initialize_session()
        chopin.load_model("saved_model/Chopin/model.ckpt")

        graph = utils.prims_initialize(x)

        ground_truth_cuts, gt_assignments = utils.generate_gt_cuts(gt, seeds, assignments=True)

        msf = self.minimum_spanning_forest(I_a, graph, seeds)

        segmentations = utils.assignments(img, msf, seeds)

        shortest_paths = nx.get_node_attributes(msf, 'path')
        assignments = nx.get_node_attributes(msf, 'seed')
        cuts = utils.get_cut_edges(msf)

        acc = utils.accuracy(assignments, gt_assignments)
        print("Accuracy: ", acc)

    @property
    def rgb_image(self):
        return self._rgb

    @property
    def padded_rgb_image(self):
        return self._padded_rgb
