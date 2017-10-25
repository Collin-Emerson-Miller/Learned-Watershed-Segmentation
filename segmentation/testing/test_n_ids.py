#!/usr/bin/python

from cremi.io import CremiFile
from cremi.evaluation import NeuronIds, Clefts, SynapticPartners

test = CremiFile('segmentations.h5', 'r')
truth = CremiFile('ground_truth.h5', 'r')

neuron_ids_evaluation = NeuronIds(truth.read_neuron_ids())

print(neuron_ids_evaluation)

(voi_split, voi_merge) = neuron_ids_evaluation.voi(test.read_neuron_ids())
adapted_rand = neuron_ids_evaluation.adapted_rand(test.read_neuron_ids())

print "Neuron IDs"
print "=========="
print "\tvoi split   : " + str(voi_split)
print "\tvoi merge   : " + str(voi_merge)
print "\tadapted RAND: " + str(adapted_rand)

print(test.read_neuron_ids())
