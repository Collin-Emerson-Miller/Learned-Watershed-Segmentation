
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import preprocessing_utils


# In[ ]:


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

