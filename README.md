# Learned-Watershed-Segmentation
This is a python implementation of the research paper "[Learned Watershed: End-to-End Learning of Seeded Segmentation](https://arxiv.org/pdf/1704.02249.pdf)"

This paper was a modification of the traditional seeded watershed segmentation algorithm in computer vision.  In its classic form, the watershed algorithm comprises three basic steps: altitude computation, seed definition, and region assignment.  This research takes the idea one step further and propose to learn altitude estimation and region assignment jointly.  To find out more on the theory, please read the paper.

This implementation is written in python and uses both Tensorflow and Keras for deep learning libraries.  An example of how to use this implementation is contained in a Jupyter notebook.  The data that is included in the example is the [MICCAI Challenge on Circuit Reconstruction from Electron Microscopy Images](https://cremi.org/). We specifically are using the cropped version of [dataset A](https://cremi.org/data/).
