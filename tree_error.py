import numpy as np
import zss
from node import Node, get_all_sequences, count_nodes
import random
import string


def asymmetric_distance(x, y, distances):
    """
    Compute the asymmetric edit distance (no deletions) between two string sequences, using precomputed EDs
    :param x: the first string (can't delete from here)
    :param y: the second string (can insert here)
    :param distances: a dict from tuples (a, b) to ED(a, b
    :return: the int AD(x, y)
    """
    while len(x) > len(y):
        y.append('')  # to get an estimate of err(T) for the errorful tree T~
        # raise RuntimeError('AD(x, y) is undefined when |x| > |y| (|x| = {}, |y| = {})'.format(len(x), len(y)))

    # dist[i][j] is the assymetric distance between x[i:] and y[j:]
    dist = [[None for _ in range(len(y) + 1)] for _ in range(len(x) + 1)]

    # Initialize matrix
    dist[len(x)][len(y)] = 0

    # Fill in bottommost row
    # '' -> b[j:] = inserting each character in y[j:]
    for j in range(len(y) - 1, -1, -1):
        dist[len(x)][j] = dist[len(x)][j + 1] + len(y[j])

    # Fill in the rest of the dist matrix right to left, bottom to top
    for i in range(len(x) - 1, -1, -1):
        for j in range(len(y) - 1, -1, -1):
            if len(x) - i == len(y) - j:
                dist[i][j] = dist[i + 1][j + 1] + distances[x[i], y[j]]  # must sub if same length
            elif len(x) - i < len(y) - j:
                insert = dist[i][j + 1] + len(y[j])
                sub = dist[i + 1][j + 1] + distances[x[i], y[j]]
                dist[i][j] = min(insert, sub)

    return dist[0][0]


def tree_error(root, sequences, alpha, distances):
    """
    Compute err(T), as defined in the paper.
    :param root: the root of T
    :param sequences: the list of sequences
    :param alpha: the node cost parameter
    :return: err(T)
    """
    node_label_seqs = get_all_sequences(root)
    if len(node_label_seqs) != len(sequences):
        return np.nan
    return alpha * count_nodes(root) + sum(asymmetric_distance(sequences[i], node_label_seqs[i], distances)
                                           for i in range(len(sequences)))


if __name__ == '__main__':
    A = Node('a')
    nodes = [A]

    for i in range(10):
        parent = random.choice(nodes)
        new_node = Node(string.ascii_lowercase[i + 1])
        parent.children.append(new_node)
        nodes.append(new_node)

    B = Node('a')
    nodes = [B]

    for i in range(5):
        parent = random.choice(nodes)
        new_node = Node(string.ascii_lowercase[i + 1])
        parent.children.append(new_node)
        nodes.append(new_node)

    print(zss.simple_distance(A, B))
