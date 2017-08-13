from __future__ import print_function

import keras
import time
import numpy as np
import tensorflow as tf
import utils


class TempNet:

    def __init__(self):
        pass

    def build(self, i_augmented):
        """
        Builds

        Args:
            i_augmented: rgb image [batch, height, width, 2]
        """

        start_time = time.time()
        print("Build model started.")

        self.i_augmented = i_augmented

        m = keras.layers.Conv2D(16, 5, padding='same',
                                activation='elu', dilation_rate=1)(i_augmented)
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
        m = keras.layers.Dense(64, activation='relu')(m)
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

        print(("Build model finished: %fs" % (time.time() - start_time)))

    def predict_edges(self, image_dict, session, max_batch_size):
        """

        Args:
            image_dict:

        Returns:

        """

        with session.as_default():

            batches = utils.create_batches(np.array(image_dict.keys()), np.array(image_dict.values()),
                                           max_batch_size=max_batch_size)

            altitudes = []

            times = []

            print("Starting predictions")
            for batch in batches:
                feed_dict = {self.i_augmented: batch[1],
                             keras.backend.learning_phase(): 0}

                start = time.time()
                altitudes.append(session.run(self.f_static, feed_dict=feed_dict))

                end = time.time()
                times.append(end - start)
            print("Prediction Done.  Average time: %fs, Total time: %fs" % (np.mean(times), np.sum(times)))

            altitudes = np.vstack(altitudes)

            altitude_dict = {}

            for i, k in enumerate(image_dict.keys()):
                altitude_dict[k] = altitudes[i][0]

        return altitude_dict