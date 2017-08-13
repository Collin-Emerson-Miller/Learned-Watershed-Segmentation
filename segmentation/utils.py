from __future__ import division

import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
import threading
import math
from heapq import heappush, heappop
from sklearn import feature_extraction


def img_to_graph(image):

    image = image.astype(np.int16)

    coo_matrix = feature_extraction.img_to_graph(image)
    graph = nx.from_scipy_sparse_matrix(coo_matrix)

    node_labels = graph.nodes()
    node_labels = np.array(node_labels)
    node_labels = node_labels.reshape(image.shape)

    mapping = {}
    for index, x in np.ndenumerate(node_labels):
        mapping[x] = index

    graph = nx.relabel_nodes(graph, mapping)
    graph.remove_edges_from(graph.selfloop_edges())

    return graph


def crop_2d(image, top_left_corner, height, width):
    """
    Returns a crop of an image.

    Args:
        image: The original image to be cropped.
        top_left_corner: The coordinates of the top left corner of the image.
        height: The hight of the crop.
        width: The width of the crop.

    Returns:
        A cropped version of the original image.
    """

    x_start = top_left_corner[0]
    y_start = top_left_corner[1]
    x_end = x_start + width
    y_end = y_start + height

    return image[x_start:x_end, y_start:y_end, ...]


def prims_initialize(graph):

    """

    Args:
        graph:

    Returns:

    """

    assignment_dict = dict()
    assignment_history = dict()

    for x in graph.nodes():
        assignment_dict[x] = 'none'
        assignment_history[x] = []

    nx.set_node_attributes(graph, "seed", assignment_dict)
    nx.set_node_attributes(graph, "path", assignment_history)

    return graph


def gradient_segmentation(graph, seeds):
    """

    Args:
        image: An image in which to generate the ground truth cuts.
        seeds: A list of seeds or starting points.

    Returns:
        A list of ground truth cuts.
    """

    nodes = graph.nodes()

    push = heappush
    pop = heappop

    print("Starting gradient segmentation...")
    start = time.time()

    while nodes:
        frontier = []
        visited = []
        for u in seeds:

            # Assign seed to self.
            graph.node[u]['seed'] = u

            nodes.remove(u)
            visited.append(u)

            # Store path.
            graph.node[u]['path'] = [u]

            # Push all edges
            for u, v in graph.edges(u):
                push(frontier, (graph[u][v].get('weight', 1), u, v))

        while frontier:
            W, u, v = pop(frontier)

            if v in visited:
                continue

            # Assign the node
            graph.node[v]['seed'] = graph.node[u]['seed']

            # Store path.
            graph.node[v]['path'] = graph.node[u]['path'] + [v]

            visited.append(v)
            nodes.remove(v)
            for v, w in graph.edges(v):
                if not w in visited:
                    push(frontier, (graph[v][w].get('weight', 1), v, w))

    end = time.time()
    print("Segmentation done: %fs" % (end - start))

    print("Calculating Cuts...")
    start = time.time()

    cuts = []

    [cuts.append(e) if graph.node[e[0]]['seed'] \
    is not graph.node[e[1]]['seed'] else '' for e in \
    graph.edges_iter()]

    end = time.time()
    print("Done: %fs" % (end - start))

    paths = nx.get_node_attributes(graph, 'path')


    return cuts, paths


def view_path(image, path):

    img = image.copy()

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for x, y in path:
        img[x, y] = [0, 0, 255]
    plt.imshow(img)


def view_boundaries(image, cuts):

    img = image.copy()

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for x, y in cuts:
        img[x[0], x[1]] = [0, 0, 255]
        img[y[0], y[1]] = [0, 255, 0]

    plt.imshow(img)


def prepare_input_images(image, height=15, width=15):
    """
    Preprocess images to be used in the prediction of the edges.

    Args:
        image (numpy.array):
    """

    npad = ((height // 2, width // 2), (height // 2, width // 2), (0, 0))
    padded_image = np.pad(image, npad, 'reflect')

    image_dict = {}

    for index in np.ndindex(image.shape[:-1]):
        image_dict[index] = crop_2d(padded_image, index, height, width)

    return image_dict


def compute_root_error_edge_children(shortest_paths, ground_truth_paths, cut_edges, ground_truth_cuts):
    """
    Computes the root error edges used for a single training epoch of the system.

    This function will prepare the weight function and the altitude prediction used for the loss.
    The approach taken here is for every node in the graph, check if the node satisfies a failure
    condition. If so, then add or subtract to the root error edge children.

    By construction of the MSF, the shortest path and the ground truth path are equal
    for all nodes.  Conversely, they differ for incorrect nodes, causing the gound truth
    path distance to exceed the shortest path distance.

    Args:
        shortest_paths: The shortest paths generated from the MSF.
        ground_truth_paths: The ground truth paths from the constrained MSF.
        cut_edges: The cut edges generated from the MSF.
        ground_truth_cuts: The cut edges generated from the constrained MSF.

    Returns:
        A dictionary in which the keys are the root error edges, and the values are the number of children
        of the edge.
    """

    start_time = time.time()
    print("Calculating Root Error Edge Children.")

    # Initialize edge error weights dictionary.
    edge_error_weights = dict()

    # Here multithreading is used to speed up root error edge computation.  Each thread
    # computes the root error edges for a node.
    threads = []
    for node, shortest_path in shortest_paths.items():
        if shortest_path != ground_truth_paths[node]:
            thread = threading.Thread(target=find_root_edge, args=[shortest_path, ground_truth_paths[node],
                                                                   cut_edges, ground_truth_cuts,
                                                                   edge_error_weights])
            threads.append(thread)
            thread.start()

            # Join threads
    [thread.join() for thread in threads]

    print(("Done: %fs" % (time.time() - start_time)))

    return edge_error_weights


def find_root_edge(shortest_path, ground_truth_path, cut_edges, ground_truth_cuts,
                   edge_error_weights):
    """
    Finds the root error edges for a node and inserts them into the dictionary.

    Args:
        ground_truth_cuts:
        shortest_path:
        ground_truth_path:
        cut_edges (list): A list of tuples representing the cuts for the graph.
        edge_error_weights (dictionary): The dictionary that holds all of the weights for
        the root error edges.
    """

    assigned_seed = shortest_path[0]
    ground_truth_seed = ground_truth_path[0]

    # Compute the root edge to increase (p(w)).
    root_missing_cut_edge = find_missing_cut(shortest_path, ground_truth_cuts,
                                             cut_edges)

    # Increment the number of children for the root edge.
    try:
        edge_error_weights[root_missing_cut_edge]

    except KeyError:

        edge_error_weights[root_missing_cut_edge] = 0
    finally:

        edge_error_weights[root_missing_cut_edge] = \
            edge_error_weights[root_missing_cut_edge] - 1

    # Compute the root edge to decrease.
    if assigned_seed != ground_truth_seed:
        root_false_cut_edge = find_first_false_cut(ground_truth_path,
                                                   ground_truth_cuts,
                                                   cut_edges)
    else:
        root_false_cut_edge = find_deviation(ground_truth_path, shortest_path)

    try:
        edge_error_weights[root_false_cut_edge]
    except KeyError:
        edge_error_weights[root_false_cut_edge] = 0
    finally:
        edge_error_weights[root_false_cut_edge] = \
            edge_error_weights[root_false_cut_edge] + 1


def find_first_false_cut(ground_truth_path, ground_truth_cuts, cut_edges):
    """
    Finds the first false cut edge of a ground truth path.


    Args:
        ground_truth_path (list): A list of nodes representing the path from the seed to the node.
        ground_truth_cuts (list): A list of ground truth cut edges.
        cut_edges (list): A list of cut edges from the minimum spanning forest.

    Returns:
        tuple: The first edge in the ground truth path that is in the list of cut edges, but not in
        in the list of ground truth edges.
    """

    for i, node in enumerate(ground_truth_path):
        try:
            edge = (ground_truth_path[i], ground_truth_path[i + 1])
            if edge in cut_edges or tuple(reversed(edge)) in cut_edges:
                if edge not in ground_truth_cuts or tuple(reversed(edge)) not in ground_truth_cuts:
                    return edge

        except IndexError:
            print "Something went wrong."
            return


def find_deviation(ground_truth_path, shortest_path):
    """
    Computes finds the edge where the ground truth path deviates from the shortest path.

    Args:
        ground_truth_path (list): The list of edges in the ground truth path.
        shortest_path (list): The list of edges in the shortest path.

    Returns:
        tuple: The first edge in which the two paths differ.
    """

    for i, (ground_truth_node, shortest_path_node) in enumerate(zip(ground_truth_path, shortest_path)):

        if shortest_path_node != ground_truth_node:
            return (ground_truth_path[i - 1], ground_truth_path[i])
    else:
        raise ValueError('No deviation.')


def find_missing_cut(shortest_path, ground_truth_cuts, cut_edges):
    """
    Computes the root error missing cut of a shortest path.

    Every incorrect shortest path has at least one erroneous cut edge.  The first such
    edge shall be called the path's root error edge p(w) and is always a missing cut.

    Args:
        shortest_path (list): The list of edges in the shortest path.
        ground_truth_cuts (list): The list ground truth cuts for the ground truth segmentation.
        cut_edges (list): The list of cut edges from the current segmentation.

    Returns:
        tuple: The first erroneous cut edge in the shortest path.
    """

    for i, node in enumerate(shortest_path):
        try:
            edge = (shortest_path[i], shortest_path[i + 1])
            if edge in ground_truth_cuts or tuple(reversed(edge)) in ground_truth_cuts:
                if edge not in cut_edges and tuple(reversed(edge)) not in cut_edges:
                    return edge

        except IndexError:
            print "Something went wrong."
            continue


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


def generate_gt_cuts(ground_truth_segmentation):
    """
    Given a numpy array containing the neuron ids for a CREMI neural segmentation image,
    this function returns the ground segmentation image where a segmentation boundary
    is a white pixel and everything else is a black pixel.

    Args:
        ground_truth_segmentation: A numpy array containing the neuron ids of a segmentation image.

    Returns:
        A numpy array of ground truth segmentations.

    """

    gt_cuts = []

    edges_right = ground_truth_segmentation[:, :-1].ravel() != ground_truth_segmentation[:, 1:].ravel()
    edges_down = ground_truth_segmentation[:-1].ravel() != ground_truth_segmentation[1:].ravel()

    edges_down = edges_down.reshape((ground_truth_segmentation.shape[0] - 1, ground_truth_segmentation.shape[1]))
    edges_right = edges_right.reshape((ground_truth_segmentation.shape[0], ground_truth_segmentation.shape[1] - 1))

    for index, x in np.ndenumerate(edges_down):
        if x:
            gt_cuts.append(((index), (index[0] + 1, index[1])))

    for index, x in np.ndenumerate(edges_right):
        if x:
            gt_cuts.append(((index), (index[0], index[1] + 1)))

    return gt_cuts






