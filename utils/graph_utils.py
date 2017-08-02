"""Utilities for creating an manipulating graphs."""

import numpy as np
import math


def create_batches(x, y, max_batch_size=32):
    """

    Args:
        x: A numpy array of the input data
        y: A numpy array of the output
        max_batch_size: The maximum elements in each batch.

    Returns: A list of batches.

    """

    batches = math.ceil(x.shape[0] / max_batch_size)
    x = np.array_split(x, batches)
    y = np.array_split(y, batches)

    return zip(x, y)
