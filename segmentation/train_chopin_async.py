from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from models import BachNet
from models import ChopinNet
from utils import display_utils
from utils import graph_utils
from utils import prediction_utils

train_path = "data/train"
test_path = "data/test"

input_path = "input"
output_path = "output"

gt_tag = "gt"

receptive_field_shape = (12, 12)
n_epochs = 5
save_rate = 10

bach = BachNet.BachNet()
bach.build(receptive_field_shape, 1)
bach.load_model('models/saved_models/Bach/model.h5')

batch = dict()
input_gen = prediction_utils.input_generator(bach, train_path, input_path, gt_tag)

while True:
    try:
        f_name, img, bps, I_a, gt, gt_cuts, seeds = next(input_gen)
        graph = graph_utils.prims_initialize(img)
        batch[f_name] = img, bps, I_a, gt, gt_cuts, seeds, graph
    except StopIteration:
        break

chopin = ChopinNet.Chopin()
chopin.build(receptive_field_shape, learning_rate=1e-5)
#chopin.load_model("models/saved_model/Chopin/checkpoint")
chopin.initialize_session()

global_loss_timeline = []
loss_timelines = dict()
loss_file = open("data/train/chopin/global_loss.txt", 'w')
loss_file.write("f_name\tepoch\tloss\n")

for epoch in range(n_epochs):
    for f_name, (img, bps, I_a, gt, gt_cuts, seeds, graph) in batch.iteritems():
        print("Training on:", f_name)

        foldername = os.path.join(train_path, "chopin", f_name)
        start = time.time()

        loss, segmentations, cuts = chopin.train_on_image(img, I_a, gt_cuts, seeds, graph)

        if epoch % save_rate:
            print("Saving Model")
            chopin.save_model(os.path.join(foldername, "saved_models", "model"), epoch)
            chopin.save_model("models/saved_models/Chopin/chopin", epoch)

        print(time.time() - start)
        print("Loss: ", loss)

        plt.imsave(os.path.join(foldername, "epoch_{}_bw.png".format(epoch)),
                   display_utils.view_boundaries(np.zeros_like(img), cuts))

        mask = display_utils.transparent_mask(display_utils.view_boundaries(img, gt_cuts), segmentations, alpha=0.75)
        plt.imsave(os.path.join(foldername, "epoch_{}_overlay.png".format(epoch)), mask)

        loss_file.write(f_name + "\t" + str(epoch) + "\t" + str(loss) + "\n")
        loss_file.flush()

        global_loss_timeline.append(loss)

        try:
            loss_timelines[f_name].append(loss)
        except KeyError:
            loss_timelines[f_name] = [loss]

        plt.plot(loss_timelines[f_name])
        plt.savefig(os.path.join(foldername, "local_loss"))

        plt.gcf().clear()

        plt.plot(global_loss_timeline)
        plt.savefig("data/train/chopin/global_loss")

        plt.gcf().clear()

loss_file.close()
