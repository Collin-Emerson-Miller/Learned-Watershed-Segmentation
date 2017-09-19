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
from heapq import heappush, heappop


class Chopin:
    def __init__(self):
        self.sess = tf.Session()

    def build(self, receptive_field_shape):

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
        # static = keras.layers.BatchNormalization()(static)
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

        self.altitude = keras.layers.Dense(16, activation='relu')(merge)
        self.altitude = keras.layers.BatchNormalization()(self.altitude)
        self.altitude = keras.layers.Dense(1, activation='elu')(self.altitude)
        self.altitude = keras.layers.BatchNormalization()(self.altitude)

        # This placeholder will hold the root error edge values.
        self.gradient_weights = tf.placeholder(tf.float32, shape=(1, None))

        # Define optimizer
        opt = tf.train.GradientDescentOptimizer(learning_rate=1e-7)

        tvs = tf.trainable_variables()

        loss = tf.matmul(self.gradient_weights, self.altitude)

        # Accumulate gradients of predictions with respect to the parameters.
        accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

        gvs = opt.compute_gradients(loss, tvs)
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

    def minimum_spanning_forest(self, I_a, graph, seeds):

        num_nodes = graph.number_of_nodes()
        visited = []
        frontier = []

        push = heappush
        pop = heappop

        relative_assignments = RelativeAssignments(seeds, (I_a.shape[0], I_a.shape[1]), self.receptive_field_shape)

        print("Starting gradient segmentation...")
        start = time.time()

        while len(visited) < num_nodes:

            for u in seeds:

                # Assign seed to self.
                graph.node[u]['seed'] = u

                relative_assignments.assign_node(u, seeds.index(u))

                visited.append(u)

                # Store path.
                graph.node[u]['path'] = [u]

                # Push all edges
                for u, v in graph.edges(u):

                    seed = graph.node[u]['seed']

                    cropped_rgb = relative_assignments.get_node_image(v, seed)

                    cropped_rgb = np.expand_dims(cropped_rgb, 0)

                    cropped_image = utils.crop_2d(I_a, v, self.receptive_field_shape[0],
                                                  self.receptive_field_shape[1])

                    cropped_image = np.expand_dims(cropped_image, 0)

                    altitude_value = self.predict_altitudes(cropped_image, cropped_rgb)

                    graph.edge[u][v]['weight'] = altitude_value[0][0]
                    try:
                        graph.edge[u][v]['static_image'] = cropped_image
                        graph.edge[u][v]['dynamic_image'] = cropped_rgb
                    except KeyError:
                        pass

                    del cropped_image, cropped_rgb

                    push(frontier, (graph[u][v].get('weight', 1), u, v))

            while frontier:
                W, u, v = pop(frontier)

                if v in visited:
                    continue

                # Assign the node
                graph.node[v]['seed'] = graph.node[u]['seed']

                relative_assignments.assign_node(v, seeds.index(graph.node[u]['seed']))

                # Store path.
                graph.node[v]['path'] = graph.node[u]['path'] + [v]

                visited.append(v)

                for v, w in graph.edges(v):
                    if not w in visited:

                        seed = graph.node[v]['seed']

                        cropped_rgb = relative_assignments.get_node_image(w, seed)

                        cropped_rgb = np.expand_dims(cropped_rgb, 0)

                        cropped_image = utils.crop_2d(I_a, w, self.receptive_field_shape[0],
                                                      self.receptive_field_shape[1])

                        cropped_image = np.expand_dims(cropped_image, 0)

                        altitude_value = self.predict_altitudes(cropped_image, cropped_rgb)

                        graph.edge[v][w]['weight'] = altitude_value[0][0]

                        try:
                            graph.edge[v][w]['static_image'] = cropped_image
                            graph.edge[v][w]['dynamic_image'] = cropped_rgb
                        except KeyError:
                            pass

                        push(frontier, (graph[v][w].get('weight', 1), v, w))

            end = time.time()
            print("Segmentation done: %fs" % (end - start))

        del relative_assignments
        del visited

        return graph


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



class RelativeAssignments:
    def __init__(self, seeds, image_size, receptive_field_shape):
        self.receptive_field_shape = receptive_field_shape
        self.seeds = seeds
        self.image_size = image_size
        self._rgb = np.zeros((len(seeds), image_size[0], image_size[1], 3))
        self._rgb[:, :, :] = [0, 0, 1]

        npad = ((0, 0), (receptive_field_shape[0] // 2, receptive_field_shape[1] // 2),
                (receptive_field_shape[0] // 2, receptive_field_shape[1] // 2), (0, 0))

        self._padded_rgb = np.pad(self._rgb, npad, 'edge')

    def assign_node(self, node, seed_index):
        x = node[0] + self.receptive_field_shape[0] // 2
        y = node[1] + self.receptive_field_shape[1] // 2

        self._padded_rgb[:seed_index, x, y] = [1, 0, 0]
        self._padded_rgb[seed_index, x, y] = [0, 1, 0]
        self._padded_rgb[seed_index + 1:, x, y] = [1, 0, 0]

        return

    def get_node_image(self, node, seed):
        seed_index = self.seeds.index(seed)

        return utils.crop_2d(self._padded_rgb[seed_index], node,
                             self.receptive_field_shape[0],
                             self.receptive_field_shape[1])

    





            # def _training_epoch(img, images_in, gt, seeds):
#     graph = utils.prims_initialize(img)
#
#     chopin = ChopinNet()
#
#     altitudes = chopin.predict_altitudes(images_in, 700)
#     altitudes = np.reshape(altitudes, img.shape)
#
#     for (x, y), d in np.ndenumerate(altitudes):
#         graph.node[(x, y)]['altitude'] = d
#         image = images_in[x + y]
#         graph.node[(x, y)]['image'] = image
#
#     print("Generating MSF")
#     msf = utils.minimum_spanning_forest(graph, seeds)
#
#     segmentations = utils.assignments(img, msf, seeds)
#
#     shortest_paths = nx.get_node_attributes(msf, 'path')
#     assignments = nx.get_node_attributes(msf, 'seed')
#     cuts = utils.get_cut_edges(msf)
#
#     print("Generating Ground Truth Graph")
#     ground_truth_cuts, gt_assignments = utils.generate_gt_cuts(gt, seeds, assignments=True)
#
#     acc = utils.accuracy(assignments, gt_assignments)
#
#     constrained_graph = graph.copy()
#
#     constrained_graph.remove_edges_from(ground_truth_cuts)
#
#     print("Generating Constrained MSF")
#     constrained_msf = utils.minimum_spanning_forest(constrained_graph, seeds)
#
#     ground_truth_paths = nx.get_node_attributes(constrained_msf, 'path')
#
#     children = utils.compute_root_error_edge_children(shortest_paths,
#                                                       ground_truth_paths, cuts,
#                                                       ground_truth_cuts)
#
#     images = []
#     weights = []
#     loss = 0
#
#     for edge, weight in children.iteritems():
#         weights.append(weight)
#         image = graph.edge[edge[0]][edge[1]]['image']
#         images.append(image)
#         altitude = graph.edge[edge[0]][edge[1]]['weight']
#
#         loss += altitude * weight
#
#     batches = utils.create_batches(np.array(images), np.array(weights))
#
#     saver = tf.train.Saver()
#
#     with tf.Session() as sess:
#
#         try:
#             # Restore variables from disk.
#             saver.restore(sess, "saved_model/Chopin/model.ckpt")
#             #print("Model restored.")
#         except:
#             sess.run(tf.global_variables_initializer())
#             save_path = saver.save(sess, "saved_model/Chopin/model.ckpt")
#             #print("Model saved in file: %s" % save_path)
#
#         # Zero out gradient accumulator.
#         sess.run(chopin.zero_ops)
#
#         # Accumulate gradients.
#         for batch in batches:
#             sess.run(chopin.accum_ops, feed_dict={chopin.i_augmented: batch[0],
#                                                   chopin.gradient_weights: [batch[1]],
#                                                   keras.backend.learning_phase(): 0})
#
#         sess.run(chopin.train_step)
#
#         save_path = saver.save(sess, "saved_model/Chopin/model.ckpt")
#         #print("Model saved in file: %s" % save_path)
#
#     return segmentations, loss, acc


# def fit(img, gt, seeds, num_epochs):
#     bach = BachNet.BachNet()
#
#     boundary_probabilities = bach.boundary_probabilities(img, verbose=1)
#
#     I_a = np.stack((img, boundary_probabilities), axis=2)
#
#     images_in = utils.prepare_input_images(I_a, height=15, width=15)
#
#     loss_timeline = []
#     segmentation_epochs = []
#     accuracy_timeline = []
#
#     for epoch in range(num_epochs):
#
#         print("\nEpoch {}".format(epoch + 1))
#         segmentations, loss, acc = _training_epoch(img, images_in, gt, seeds)
#         loss_timeline.append(loss)
#         segmentation_epochs.append(segmentations)
#         accuracy_timeline.append(acc)
#
#     return boundary_probabilities, segmentation_epochs, loss_timeline, accuracy_timeline
#
#
# def predict(img, seeds, receptive_field_shape):
#     bach = BachNet.BachNet()
#
#     boundary_probabilities = bach.boundary_probabilities(img, verbose=1)
#
#     I_a = np.stack((img, boundary_probabilities), axis=2)
#
#     images_in = utils.prepare_input_images(I_a, height=receptive_field_shape[0],
#                                            width=receptive_field_shape[1])
#
#     graph = utils.prims_initialize(img)
#
#     chopin = ChopinNet()
#
#     altitudes = chopin.predict_altitudes(images_in, 700)
#     altitudes = np.reshape(altitudes, img.shape)
#
#     for (x, y), d in np.ndenumerate(altitudes):
#         graph.node[(x, y)]['altitude'] = d
#         image = images_in[x + y]
#         graph.node[(x, y)]['image'] = image
#
#     print("Generating MSF")
#     msf = utils.minimum_spanning_forest(graph, seeds)
#
#     segmentations = utils.assignments(img, msf, seeds)
#
#     return segmentations
