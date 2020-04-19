import sys
from glob import glob

import networkx as nx
from editdistance import eval as ed

from all_distances import get_distances_sequential
from node import Node, get_leaves, find_medoid

ED_THRESHOLD = 5
ROOT = 'suzannedathegrenoblefrance'


def pnas_build_tree(sequences, threshold=ED_THRESHOLD):
    successors = dict()
    equiv = dict()
    last_nodes = set()

    first_labels = [[seq[0]] for seq in sequences]
    root_label = find_medoid([x[0] for x in first_labels], get_distances_sequential(first_labels))

    for seq in sequences:
        if ed(seq[0], root_label) <= threshold:
            equiv[seq[0]] = root_label

        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i+1]

            # replace a with its equivalent representative
            if a in equiv:
                a = equiv[a]
            else:
                equiv[a] = a

            if a not in successors:
                successors[a] = dict()

            # if b is close enough to another successor of a, then use that as b's equivalent representative
            if b not in equiv:
                equiv[b] = b
            for c in successors[a]:
                if ed(b, c) <= threshold:
                    equiv[b] = c
                    b = c

            if b in successors[a]:
                successors[a][b] += 1
            else:
                successors[a][b] = 1

        last_nodes.add(equiv[seq[-1]])

    G = nx.DiGraph()
    for name in successors:
        for successor, weight in successors[name].items():
            G.add_edge(name, successor, weight=weight)

    remove = []
    removed_node = True
    while removed_node:
        removed_node = False
        for node, in_degree in G.in_degree():
            if in_degree == 0 and node != root_label:
                removed_node = True
                remove.append(node)
        G.remove_nodes_from(remove)

    try:
        T = nx.maximum_spanning_arborescence(G)
    except nx.exception.NetworkXException:
        print('No spanning tree')
        return None

    paths = nx.single_source_shortest_path(T, root_label)
    keep = [node for last_node in last_nodes if last_node in paths for node in paths[last_node]]
    remove = set(T.nodes()).difference(set(keep))

    T.remove_nodes_from(remove)

    last_string_indices = dict()
    for i, seq in enumerate(sequences):
        last_string_indices[seq[-1]] = i

    root = to_node_tree(T)
    for leaf in get_leaves(root):
        if leaf.label in last_string_indices:
            leaf.petition_index = last_string_indices[leaf.label]
        else:
            leaf.petition_index = 0
    root.sort_by_inversions()

    return root


def to_node_tree(T, node=None):
    if node is None:
        root = next(n for n, d in T.in_degree() if d == 0)
        node = Node(root)

    for child in T.neighbors(node.label):
        child_node = Node(child)
        node.add_child(child_node)
        to_node_tree(T, child_node)

    return node


def pnas_build_tree_from_files(in_dir):
    sequences = []
    for file in glob(in_dir + '/*.txt'):
        with open(file) as f:
            sequences.append(f.read().strip().split('\n'))

    return pnas_build_tree(sequences)


if __name__ == '__main__':
    pnas_build_tree_from_files(sys.argv[1])





