
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from . import preprocessing_utils


class RelativeAssignments:
    """Container for relative node assignments.
        
        Args:
            seeds: A `list` of seeds `tuples`.
            image_size: An image shape `tuple`.
            receptive_field_shape: The shape of the crops to be input to the network.   
            
        >>> seeds = [(0, 0), (50, 50)]
        >>> img = np.random.rand(100, 100)
        >>> rfs = (23, 23)
        >>> ra = Relative_assignments(seeds, img.shape, rfs)
        """
    def __init__(self, seeds, image_size, receptive_field_shape):
        self.receptive_field_shape = receptive_field_shape
        self.seeds = seeds
        self.image_size = image_size
        self._rel_assign = np.zeros((len(seeds), image_size[0], image_size[1], 3))
        self._rel_assign[:, :, :] = [0, 0, 1]

        npad = ((0, 0), (receptive_field_shape[0] // 2, receptive_field_shape[1] // 2),
                (receptive_field_shape[0] // 2, receptive_field_shape[1] // 2), (0, 0))

        self._p_rel_assign = np.pad(self._rel_assign, npad, 'edge')

    def assign_node(self, node, seed_index):
        x = node[0] + self.receptive_field_shape[0] // 2
        y = node[1] + self.receptive_field_shape[1] // 2

        self._p_rel_assign[:seed_index, x, y] = [1, 0, 0]
        self._p_rel_assign[seed_index, x, y] = [0, 1, 0]
        self._p_rel_assign[seed_index + 1:, x, y] = [1, 0, 0]

        return
    
    
    def prepare_images(self, prepared_input):
        """Returns the input images for the given nodes.
        
        Args:
            
            
        Returns:
            A `ndarray` of input images for the network.
        """
        
        images = []
        
        for node, seed_index in prepared_input:
            images.append(preprocessing_utils.crop_2d(self._p_rel_assign[seed_index], node,
                                        self.receptive_field_shape[0],
                                        self.receptive_field_shape[1]))
        return np.stack(images)
        
    @property
    def padded_ras(self):
        return self._p_rel_assign
