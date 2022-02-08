# code adapted from Kristen M. Altenburger, which is adapted from Sam F. Way

from scipy.io import loadmat
from numpy import genfromtxt
import networkx as nx
import numpy as np


## function to create + save dictionary of features
def create_dict(key, obj):
    return (dict([(key[i], obj[i]) for i in range(len(key))]))


def parse_fb100_mat_file(filename):
    """ Parse FB100 .mat files.

        Metadata values:
            0 - student/faculty status
            1 - gender
            2 - major
            3 - second major/minor
            4 - dorm/house
            5 - year
            6 - high school
        ** Missing data coded as 0 **

        Parameters:
        - filename = path to FB100 .mat file

        Returns:
        (adj_matrix, metadata)
        - adj_matrix = adjacency matrix for the network
        - metadata = matrix containing metadata for each node

    """
    mat = loadmat(filename)
    error_msg = "%s is not a valid FB100 .mat file.  Must " \
                "contain data for variable '%s'"

    if 'A' not in mat:
        raise ValueError(error_msg % (filename, 'A'))
    adj_matrix = mat['A']

    if 'local_info' not in mat:
        raise ValueError(error_msg % (filename, 'local_info'))
    metadata = mat['local_info']

    return adj_matrix, metadata


def parse_mixing_matrix_file(filename):
    """ Parse mixing matrix made with make_mixing_matrices.py

        Returns a numpy matrix and a key describing the original
        metadata value associated with each row/column index.
    """
    handle = open(filename, 'rU')
    header = handle.readline().strip()
    if header[0] != '#':
        raise ValueError('%s has an unexpected format' % filename)
    header = header[1:]
    keys = [int(x) for x in header.split(',')]
    handle.close()

    M = genfromtxt(filename, delimiter=',')

    return M, keys


def create_graph(adj_matrix_tmp, dictionary, attribute, val_to_drop, directed_type, delete_na_cols):
    """ create graph from adjacency matrix with gender information, 
        delete nodes without gender information, keep only the largest connected component

    Parameters:
        - adj_matrix_tmp: an adjacency matrix for the graph
        - attribute: the property that we want to set for each node
        - dictionary: (key, value) pairs for each node
        - val_to_drop: value that needs to be dropped
        - attribute: which property is considered
        
    return:
        - a graph with attribute information for each node
    """
    graph = nx.from_numpy_matrix(adj_matrix_tmp)
    # get the label for each person
    keys = np.array(range(len(dictionary.keys())))
    ## we relabel keys but this preserves corresponding value with updated keys
    y_vector_list = list(dictionary.values())
    # create an adjacency matrix with the rows and columns are ordered according to the nodes in dictionary.keys()
    adj_matrix_input = nx.adj_matrix(graph,
                                     nodelist=dictionary.keys()).todense()  # note: will automatically be an out-link matrix when graph is directed
    # set each node's attribute to be the key
    nx.set_node_attributes(graph, dictionary, attribute)

    ### remove NA labeled nodes and get new graph(with all nodes have gender label)
    keys_new = []
    for i in range(len(y_vector_list)):
        if y_vector_list[i] != val_to_drop:
            keys_new.append(keys[i])
    # convert new keys into an array
    keys = np.array(keys_new)
    # delete all 0 elements in the y_vector_list
    y_vector_list = [c for c in y_vector_list if c != val_to_drop]
    # convert y_vector_list into an array
    y_vector_list = np.array(y_vector_list)
    adj_matrix_input = adj_matrix_input[np.array(keys), :]
    # and remove NA nodes in column too
    if delete_na_cols == 'yes':
        # update the graph and adjacency matrix
        adj_matrix_input = adj_matrix_input[:, np.array(keys)]
        graph = nx.from_numpy_matrix(adj_matrix_input)
        attr_new = create_dict(range(adj_matrix_input.shape[0]), y_vector_list)
        nx.set_node_attributes(graph, attr_new, attribute)

    ## create undirected network, subset to nodes only in largest connected component
    if directed_type == None:
        if nx.number_connected_components(graph) > 1:
            # print(nx.number_connected_components(graph))
            max_cc_index = 0
            max_cc_size = 0
            for c in nx.connected_components(graph):
                if graph.subgraph(c).size() > max_cc_size:
                    max_cc_index = c
                    max_cc_size = graph.subgraph(c).size()
            graph_new = graph.subgraph(max_cc_index)
            graph = graph_new

    # get the gender vector for each node
    gender_vector = nx.get_node_attributes(graph, attribute)
    gender_y = list(gender_vector.values())
    gender_y = np.array(gender_y)
    return (graph, gender_y)
