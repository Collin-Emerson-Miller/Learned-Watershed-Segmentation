from __future__ import print_function
from __future__ import division

import time
from heapq import heappop as pop
from heapq import heappush as push

import keras
import networkx as nx
import numpy as np
import tensorflow as tf
from collections import Counter
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

    def save_model(self, filepath, global_step):
        saver = tf.train.Saver()

        with self.sess.as_default():
            save_path = saver.save(self.sess, filepath, global_step)

    def predicted_msf(self, I_a, graph, seeds):
        n_visited = 0
        msf = nx.Graph()
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

        for s in seeds:

            # Add node to MSF.
            msf.add_node(s)

            # Assign seed to itself.
            msf.node[s]['seed'] = s
            ra.assign_node(s, seeds.index(s))
            n_visited += 1
            print(n_visited / visited.size)

            visited[s[0], s[1]] = 1

            # Push all edges
            for u, v in graph.edges(s):
                seed_index = seeds.index(msf.node[u]['seed'])
                static_image = static_input_images[v[0] * I_a.shape[1] + v[1]]
                dynamic_image = ra.prepare_images([(v, seed_index)])[0]
                static_image = np.expand_dims(static_image, 0)
                dynamic_image = np.expand_dims(dynamic_image, 0)
                altitude_value = self.predict_altitudes(static_image,
                                                          dynamic_image)
                graph.edge[u][v]['static_image'] = static_image
                graph.edge[u][v]['dynamic_image'] = dynamic_image
                graph.edge[u][v]['weight'] = altitude_value

                push(frontier, (graph.edge[u][v]['weight'], u, v))

        while frontier:
            W, u, v = pop(frontier)

            # If the node is already visited, then skip assigning it.
            if visited[v[0], v[1]] == 1:
                continue

            msf.add_node(v)

            # Add edge to MSF.
            msf.add_edge(u, v, graph.get_edge_data(u, v))

            # Assign the node
            msf.node[v]['seed'] = msf.node[u]['seed']
            ra.assign_node(v, seeds.index(msf.node[u]['seed']))

            # Mark as visited
            visited[v[0], v[1]] = 1

            # Increment the number of visited nodes
            if n_visited % 100 == 0:
                n_visited += 1
                print(n_visited, visited.size, n_visited / visited.size)

            for v, w in graph.edges(v):
                if visited[w[0], w[1]] == 0:
                    # Calculate the altitude of the edge.
                    seed_index = seeds.index(msf.node[v]['seed'])
                    static_image = static_input_images[w[0] * I_a.shape[1] + w[1]]
                    dynamic_image = ra.prepare_images([(w, seed_index)])[0]
                    static_image = np.expand_dims(static_image, 0)
                    dynamic_image = np.expand_dims(dynamic_image, 0)
                    altitude_value = self.predict_altitudes(static_image,
                                                              dynamic_image)
                    graph.edge[v][w]['static_image'] = static_image
                    graph.edge[v][w]['dynamic_image'] = dynamic_image
                    graph.edge[v][w]['weight'] = altitude_value
                    push(frontier, (altitude_value, v, w))

        end = time.time()
        print("Segmentation done: %fs" % (end - start))

        return msf

    def constrained_msf(self, I_a, graph, msf, seeds, gt_cuts):

        constrained_msf = nx.Graph()

        visited = np.zeros(I_a.shape[:-1])
        frontier = []

        print("Starting gradient segmentation...")
        start = time.time()

        for s in seeds:

            # Add node to MSF.
            constrained_msf.add_node(s)

            # Assign seed to itself.
            constrained_msf.node[s]['seed'] = s

            visited[s[0], s[1]] = 1

            # Push all edges
            for u, v in graph.edges(s):
                if (u, v) not in gt_cuts:
                    push(frontier, (graph.edge[u][v]['weight'], u, v))

        while frontier:
            W, u, v = pop(frontier)

            # If the node is already visited, then skip assigning it.
            if visited[v[0], v[1]] == 1:
                continue

            constrained_msf.add_node(v)

            # Add edge to MSF.msf
            constrained_msf.add_edge(u, v, graph.get_edge_data(u, v))

            # Assign the node
            constrained_msf.node[v]['seed'] = constrained_msf.node[u]['seed']

            visited[v[0], v[1]] = 1

            for v, w in graph.edges(v):
                if visited[w[0], w[1]] == 0:
                    if (v, w) not in gt_cuts and (w, v) not in gt_cuts:
                        push(frontier, (graph.edge[v][w]['weight'], v, w))

        end = time.time()
        print("Segmentation done: %fs" % (end - start))

        return constrained_msf
    
    def train_on_image(self, img, I_a, gt_cuts, seeds, graph):
        msf = self.predicted_msf(I_a, graph, seeds)
        cuts = graph_utils.get_cut_edges(graph, msf)
        constrained_msf = self.constrained_msf(I_a, graph, msf, seeds, gt_cuts)
        shortest_paths, ground_truth_paths = graph_utils.get_paths(graph, msf, constrained_msf)

        children = graph_utils.compute_root_error_edge_children(shortest_paths,
                                                                ground_truth_paths, cuts,
                                                                gt_cuts)

        segmentations = display_utils.assignments(np.zeros_like(img), msf, seeds)

        weights = []
        static_images = []
        dynamic_images = []

        for (u, v), weight in children.iteritems():
            static_images.append(graph.get_edge_data(u, v)['static_image'])
            dynamic_images.append(graph.get_edge_data(u, v)['dynamic_image'])
            weights.append(weight)
            altitude_val = graph.get_edge_data(u, v)['weight']

        batches = zip(preprocessing_utils.create_batches(np.expand_dims(np.stack(weights), 1)),
                      preprocessing_utils.create_batches(np.concatenate(static_images)),
                      preprocessing_utils.create_batches(np.concatenate(dynamic_images)))

        loss = 0
        with self.sess.as_default():
            self.sess.run(self.zero_ops)

            for w, s, d in batches:
                feed_dict = {self.gradient_weights: w.transpose(),
                             self.static_input: s,
                             self.dynamic_input: d,
                             keras.backend.learning_phase(): 0}

                self.sess.run(
                    self.accum_ops, feed_dict)
                loss += self.sess.run(self.loss, feed_dict)[0][0]

            self.sess.run(self.train_step)

        return loss, segmentations, cuts
