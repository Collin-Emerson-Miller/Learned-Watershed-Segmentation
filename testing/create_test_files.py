# Crafted by Collin Miller

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cremi
import numpy as np
from segmentation import learnedwatershed as lw

gt = cremi.CremiFile("ground_truth.h5", "w")
segs = cremi.CremiFile("segmentations.h5", "w")

segmentations = np.ones((10, 100, 10), dtype=np.uint8)
ground_truths = np.ones((10, 100, 10), dtype=np.uint8)

neuron_ids = cremi.Volume(segmentations)
gt_ids = cremi.Volume(ground_truths)

gt.write_neuron_ids(gt_ids)
segs.write_neuron_ids(neuron_ids)

gt.close()
segs.close()

print(segmentations)
print(ground_truths)