import numpy as np
import utils
from keras.models import Sequential, load_model
from keras.layers import Conv2D
from keras.layers import Flatten, Dense
from keras import backend as K

class BachNet:

    def __init__(self):
        pass

    def build(self, width, height, channels,  first_layer_num_of_filters=32, first_layer_kernel_size=(5, 5),
              first_layer_strides=3, second_layer_num_of_filters=256, second_layer_kernel_size=(3, 3),
              second_layer_strides=2):

        self.receptive_field_shape = (width, height)

        self.model = Sequential()

        # Layer 1
        self.model.add(
            Conv2D(filters=first_layer_num_of_filters, kernel_size=first_layer_kernel_size,
                   strides=first_layer_strides, padding='valid', name="Conv_1",
                   activation='relu', input_shape=(width, height, channels)))

        # Layer 2
        self.model.add(Conv2D(filters=second_layer_num_of_filters, kernel_size=second_layer_kernel_size,
                         strides=second_layer_strides, padding='valid', name="Conv_2",
                         activation='relu'))

        # Layer 3
        self.model.add(Flatten(name='Flatten_1'))
        self.model.add(Dense(512, activation='sigmoid', name='Dense_1'))
        self.model.add(Dense(1, activation='sigmoid', name='Dense_2'))

        # categorical_crossentropy
        # sparse_categorical_crossentropy
        self.model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

    def boundary_probabilities(self, image, batch_size=32, width=23, height=23, verbose=0):

        images = utils.prepare_input_images(image, width=width, height=height)

        probabilities = self.model.predict(images, batch_size=batch_size,
                                           verbose=verbose)

        probabilities = np.reshape(probabilities, image.shape)

        return probabilities

    def load_model(self):
        try:
            self.model = load_model('saved_model/Bach/model.h5')
        except IOError:
            print("Could not find model. Creating a new one.")
            self.build(self.receptive_field_shape[0], self.receptive_field_shape[1], 1)
            self.model.save('saved_model/Bach/model.h5')
