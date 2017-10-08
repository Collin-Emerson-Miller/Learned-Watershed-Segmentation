
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import networkx as nx
import numpy as np
import os
import time

from heapq import heappush, heappop
from . import preprocessing_utils


# In[2]:


def boundary_probabilities(bach, image, batch_size=32, width=23, height=23, verbose=0):

    images = preprocessing_utils.prepare_input_images(image, width=width, height=height)

    probabilities = bach.model.predict(images, batch_size=batch_size,
                                       verbose=verbose)

    probabilities = np.reshape(probabilities, image.shape)

    return probabilities

def minimum_spanning_forest(chopin, I_a, graph, seeds):

        num_nodes = graph.number_of_nodes()
        visited = []
        frontier = []

        push = heappush
        pop = heappop

        relative_assignments = RelativeAssignments(seeds,
                                                   (I_a.shape[0],
                                                    I_a.shape[1]),
                                                   chopin.receptive_field_shape)

        print("Starting gradient segmentation...")
        start = time.time()

        while len(visited) < num_nodes:

            for u in seeds:

                # Assign seed to chopin.
                graph.node[u]['seed'] = u

                relative_assignments.assign_node(u, seeds.index(u))

                visited.append(u)

                # Store path.
                graph.node[u]['path'] = [u]

                # Push all edges
                for u, v in graph.edges(u):

                    seed = graph.node[u]['seed']

                    cropped_rgb = relative_assignments.get_node_image(v,
                                                                      seed)

                    cropped_rgb = np.expand_dims(cropped_rgb, 0)

                    cropped_image = preprocessing_utils.crop_2d(I_a,
                                                  v,
                                                  chopin.receptive_field_shape[0],
                                                  chopin.receptive_field_shape[1])

                    cropped_image = np.expand_dims(cropped_image, 0)

                    altitude_value = chopin.predict_altitudes(cropped_image,
                                                              cropped_rgb)

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

                relative_assignments.assign_node(v,
                                                 seeds.index(graph.node[u]['seed']))

                # Store path.
                graph.node[v]['path'] = graph.node[u]['path'] + [v]

                visited.append(v)

                for v, w in graph.edges(v):
                    if not w in visited:

                        seed = graph.node[v]['seed']

                        cropped_rgb = relative_assignments.get_node_image(w, seed)

                        cropped_rgb = np.expand_dims(cropped_rgb, 0)

                        cropped_image = preprocessing_utils.crop_2d(I_a,
                                                      w,
                                                      chopin.receptive_field_shape[0],
                                                      chopin.receptive_field_shape[1])

                        cropped_image = np.expand_dims(cropped_image, 0)

                        altitude_value = chopin.predict_altitudes(cropped_image,
                                                                  cropped_rgb)

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

	gc.collect()

        return graph

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

        return preprocessing_utils.crop_2d(self._padded_rgb[seed_index], node,
                             self.receptive_field_shape[0],
                             self.receptive_field_shape[1])
