from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



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
        opt = tf.train.AdamOptimizer(1e-7)

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


    def fit(self, img, I_a, gt, seeds, foldername, num_epochs=8):

        I_a = utils.pad_for_window(I_a, self.receptive_field_shape[0], self.receptive_field_shape[1])
        graph = utils.prims_initialize(img)

        ground_truth_cuts, gt_assignments = utils.generate_gt_cuts(gt, seeds, assignments=True)

        loss_timeline = []
        acc_timeline = []

        for epoch in range(num_epochs):
            #print("Epoch: ", epoch)

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

            constrained_msf = utils.minimum_spanning_forest(constrained_msf, seeds)

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

            with self.sess.as_default():
                self.sess.run(self.zero_ops)

                for w, s, d in batches:
                    feed_dict = {self.gradient_weights: w.transpose(),
                                 self.static_input: s,
                                 self.dynamic_input: d,
                                 keras.backend.learning_phase(): 0}

                    self.sess.run(self.accum_ops, feed_dict)

                self.sess.run(self.train_step)



            loss_timeline.append(loss)
            print("Loss: ", loss)

            f, axarr = plt.subplots(2, sharex=True)
            axarr[0].plot(loss_timeline)
            axarr[0].set_title("Loss")
            axarr[1].plot(acc_timeline)
            axarr[1].set_title("Accuracy")

            figurname = "loss_and_accuracy"

            plt.savefig(os.path.join(foldername, figurname))

            self.save_model("saved_model/Chopin/model.ckpt")
        gc.collect()

        return segmentations, loss_timeline, acc_timeline

    def evaluate(self, images, augmented_images, ground_truth_images, seeds):

        segmentations = []
        accuracies = []

        for img, I_a, gt, s in zip(images, augmented_images, ground_truth_images, seeds):
            ground_truth_cuts, gt_assignments = utils.generate_gt_cuts(gt,
                                                                       s,
                                                                       assignments=True)

            graph = utils.prims_initialize(img)

            msf = self.minimum_spanning_forest(I_a, graph, s)

            segmentations.append(utils.assignments(img, msf, s))

            assignments = nx.get_node_attributes(msf, 'seed')

            acc = utils.accuracy(assignments, gt_assignments)

            accuracies.append(acc)

        return segmentations, accuracies

    def segment(self, images, augmented_images, seeds):

        graph = utils.prims_initialize(images)
        msf = self.minimum_spanning_forest(augmented_images, graph, seeds)
        segmentations = utils.assignments(images, msf, seeds)

        return segmentations


    @property
    def rgb_image(self):
        return self._rgb

    @property
    def padded_rgb_image(self):
        return self._padded_rgb





