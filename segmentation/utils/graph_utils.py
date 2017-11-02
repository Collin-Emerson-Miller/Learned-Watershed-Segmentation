from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import networkx as nx
import threading
import time


from heapq import heappush as push
from heapq import heappop as pop
from sklearn import feature_extraction




def img_to_graph(image):
    """Converts an image to a graph.
    
    This function takes in an image and returns a 
    4-connected grid graph.  The nodes of this graph
    are labeled as such: every pixel is a node, 
    the label of each node is the corresponding 
    (x, y) coordinates.
    
    Args:
        image (numpy_array): The input image.
        
    Returns:
        A network X graph.
    
    """

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


# In[ ]:


def prims_initialize(img):
    """Initializes an image for prims algorithm.
    
    This function takes in an image and returns
    a graph.  Each node in this graph will have a
    label, an assigned seed variable to be used
    in the minimum spanning forest, and the path
    from the assigned seed to the respective node.

    Args:
        img (numpy_array):  The image to be initialized.

    Returns:
        An initialized 4-connected grid graph.
    """

    graph = img_to_graph(img)

    assignment_dict = dict()
    assignment_history = dict()

    for x in graph.nodes():
        assignment_dict[x] = 'none'
        assignment_history[x] = []

    nx.set_node_attributes(graph, "seed", assignment_dict)
    nx.set_node_attributes(graph, "path", assignment_history)

    return graph


# In[ ]:


def minimum_spanning_forest(img, graph, seeds, timed=False):
    """Computes the minimum spanning forest for an image.
    
    This function computes the minimum spanning forest 
    for an image.  The weights for the graph are the 
    pixel gradients of the image.  Starting from the
    given seeds, each region is grown until the entire
    image is segmented.
    
    Args:
        graph (nx_graph): A networkx graph that has 
        been initialized.
        seeds (list): A list of (x, y) tuples to start 
        region growing.  
        timed (boolean): A flag that if True, will display
        how long it took to run the minimum spanning forest.
        
    Returns:
        A networkx graph with every node assigned to a 
        seed and the path from each seed to their respective 
        node.
    
    """
    
    
    num_nodes = graph.number_of_nodes()
    visited = np.zeros(img.shape)
    frontier = []

    if timed:         
        print("Starting gradient segmentation...")
        start = time.time()

    for u in seeds:

        # Assign seed to self.
        graph.node[u]['seed'] = u

        visited[u[0], u[1]] = 1

        # Store path.
        graph.node[u]['path'] = [u]

        # Push all edges
        for u, v in graph.edges(u):
            push(frontier, (graph[u][v].get('weight', 1), u, v))

    while frontier:
        W, u, v = pop(frontier)

        if visited[v[0], v[1]] == 1:
            continue

        # Assign the node
        graph.node[v]['seed'] = graph.node[u]['seed']

        # Store path.
        graph.node[v]['path'] = graph.node[u]['path'] + [v]

        visited[v[0], v[1]] = 1

        for v, w in graph.edges(v):
            if visited[w[0], w[1]] == 0:
                push(frontier, (graph[v][w].get('weight', 1), v, w))

    if timed:
        end = time.time()
        print("Segmentation done: %fs" % (end - start))

    return graph

# In[ ]:


def compute_root_error_edge_children(shortest_paths,
                                     ground_truth_paths,
                                     cut_edges,
                                     ground_truth_cuts,
                                     timed=False):
    """Computes the root error edges used for a single
    training epoch of the system.

    This function will prepare the weight function and 
    the altitude prediction used for the loss.  The approach
    taken here is for every node in the graph, check if the
    node satisfies a failure condition. If so, then 
    add or subtract to the root error edge children.

    By construction of the MSF, the shortest path and
    the ground truth path are equal for all nodes.  
    Conversely, they differ for incorrect nodes, 
    causing the gound truth path distance to exceed
    the shortest path distance.

    Args:
        shortest_paths: The shortest paths generated 
        from the MSF.
        ground_truth_paths: The ground truth paths from 
        the constrained MSF.
        cut_edges: The cut edges generated from the MSF.
        ground_truth_cuts: The cut edges generated from
        the constrained MSF.

    Returns:
        A dictionary in which the keys are the root error edges,
        and the values are the number of children of the edge.
    """

    if timed:
        start_time = time.time()
        print("Calculating Root Error Edge Children.")

    # Initialize edge error weights dictionary.
    edge_error_weights = dict()

    # Here multithreading is used to speed up root error edge 
    # computation.  Each thread computes the root error edges
    # for a node.
    threads = []
    
    for node, shortest_path in shortest_paths.items():
        
        if shortest_path != ground_truth_paths[node]:
            
            thread = threading.Thread(target=find_root_edge,
                                      args=[shortest_path,
                                            ground_truth_paths[node],
                                            cut_edges, ground_truth_cuts,
                                            edge_error_weights])
            threads.append(thread)
            thread.start()
            
    # Join threads
    [thread.join() for thread in threads]

    if timed:
        print(("Done: %fs" % (time.time() - start_time)))

    return edge_error_weights


# In[ ]:


def find_root_edge(shortest_path,
                   ground_truth_path,
                   cut_edges,
                   ground_truth_cuts,
                   edge_error_weights):
    """Finds the root error edges for a node
    and inserts them into the dictionary.

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
    root_missing_cut_edge = find_missing_cut(shortest_path, ground_truth_cuts)

    # Increment the number of children for the root edge.
    try:
        edge_error_weights[root_missing_cut_edge]

    except KeyError:

        edge_error_weights[root_missing_cut_edge] = 0
    finally:

        edge_error_weights[root_missing_cut_edge] =             edge_error_weights[root_missing_cut_edge] - 1

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
        edge_error_weights[root_false_cut_edge] =             edge_error_weights[root_false_cut_edge] + 1


# In[ ]:


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
            print("Error: No false cut.")
            return


# In[ ]:


def find_missing_cut(shortest_path, ground_truth_cuts):
    """
    Computes the root error missing cut of a shortest path.

    Every incorrect shortest path has at least one erroneous cut edge.  The first such
    edge shall be called the path's root error edge p(w) and is always a missing cut.

    Args:
        shortest_path (list): The list of edges in the shortest path.
        ground_truth_cuts (list): The list ground truth cuts for the ground truth segmentation.

    Returns:
        tuple: The first erroneous cut edge in the shortest path.
    """

    for i, node in enumerate(xrange(len(shortest_path) - 1)):
        u = shortest_path[i]
        v = shortest_path[i+1]
        if (u, v) in ground_truth_cuts or (v, u) in ground_truth_cuts:
                return (u, v)

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

# In[ ]:


def get_cut_edges(graph):
    cuts = []

    for u, v in graph.edges_iter():
        if graph.node[u]['seed'] is not graph.node[v]['seed']:
            cuts.append((u, v))

    return cuts


# In[ ]:


def accuracy(assignments, gt_assignments):
    correct = 0

    for k, v in assignments.iteritems():
        if v == gt_assignments[k]:
            correct += 1

    return correct / len(assignments)

