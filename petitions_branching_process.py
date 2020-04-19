import pickle
import random
import string
import sys
from collections import deque
from queue import Queue

import numpy as np

from node import Node, get_all_sequences, get_all_histories, find_medoid, get_leaves

ALPHABET = string.ascii_lowercase
ALPHABET_SIZE = len(ALPHABET)

P0 = 0.03  # termination probability
P2 = 0.03  # divergence probability

P_WORD_SUB = 0.001
P_WORD_DEL = 0.001

P_CHAR_SUB = 0.01
P_CHAR_DEL = 0.01

WORD_LENGTH = 25
NUM_TRIALS = 100


def get_random_string():
    """
    Generate a random node label
    :return: a string
    """
    word = ''
    for i in range(WORD_LENGTH):
        word += random.choice(ALPHABET)
    return word
    # return namegenerator.gen()


def generate_propagation_tree(p2, p0):
    """
    Create a tree using a branching process
    :param p2: divergence probability
    :param p0: termination probalility
    :return: the root of the tree, as a Node
    """
    root = Node('ROOT', is_sentinel=True)
    active_nodes = Queue()
    active_nodes.put(root)
    num_sequences = 1

    while not active_nodes.empty():
        if num_sequences > 300:
            return Node('oops')
        current = active_nodes.get()
        rand = random.random()
        if rand <= p2:
            # Diverge!
            child1 = Node(get_random_string())
            child2 = Node(get_random_string())
            current.add_child(child1, child2)
            active_nodes.put(child1)
            active_nodes.put(child2)
            num_sequences += 1
        elif rand > p0 + p2:
            # Extend!
            child = Node(get_random_string())
            current.add_child(child)
            active_nodes.put(child)
        else:  # Terminate
            current.add_child(Node('LEAF', is_sentinel=True))

    return root


def make_corrupted_histories(root, word_sub=P_WORD_SUB, word_del=P_WORD_DEL, char_sub=P_CHAR_SUB, char_del=P_CHAR_DEL):
    """
    Add string-level errors that inherit down the tree and character-level errors that occur once
    :param root: a Node
    :param word_sub: the probability that a word is substituted at each step
    :param word_del: the probability that a word is deleted at each step
    :param char_sub: the probability that each character is substituted
    :param char_del: the probability that each character is deleted
    """
    active_nodes = Queue()
    active_nodes.put(root)

    while not active_nodes.empty():
        current = active_nodes.get()
        if current.parent is not None:
            sub_indexes = np.random.choice(np.arange(len(current.history)),
                                           np.random.binomial(len(current.history), word_sub), replace=False)
            for i in sub_indexes:
                if current.history[i][0] != '':
                    current.history[i] = get_random_string(), current.history[i][1]
            del_indexes = np.random.choice(np.arange(len(current.history)),
                                           np.random.binomial(len(current.history), word_del), replace=False)
            del_indexes[::-1].sort()
            for i in del_indexes:
                current.history[i] = '', current.history[i][1]

        if not current.is_sentinel:
            current.history.append((corrupt_string(current.label, char_sub, char_del), current))

        if len(current.children) == 1:
            current.children[0].history = current.history
        else:
            for child in current.children:
                child.history = list(current.history)
        for child in current.children:
            active_nodes.put(child)


def corrupt_string(x, char_sub, char_del):
    """
    Add char-level errors to a string
    :param x: the string
    :param char_sub: prob of substituting a character
    :param char_del: prob of deleting a character
    :return: the corrupted string
    """
    if x == '':
        return x

    sub_indices = np.random.choice(np.arange(len(x)), np.random.binomial(len(x), char_sub), replace=False)
    for i in sub_indices:
        new_char = random.choice(ALPHABET)
        x = x[:i] + new_char + x[i + 1:]

    del_indexes = np.random.choice(np.arange(len(x)), np.random.binomial(len(x), char_del), replace=False)
    del_indexes[::-1].sort()
    for i in del_indexes:
        x = x[:i] + x[i + 1:]

    return x


def get_corrupted_tree(root, distances):
    """
    Make a new tree from the given root, using the leaf histories to pick labels. Represents a reconstruction with
    correct topology, but with error in labels
    :param root: a Node
    :return a Node root of the corrupted tree
    """
    node_labels = dict()

    for leaf in get_leaves(root):
        for label, node in leaf.history:
            if node not in node_labels:
                node_labels[node] = []
            if label != '':
                node_labels[node].append(label)

    node_labels = {node: find_medoid(node_labels[node], distances) for node in node_labels}

    new_root = Node('ROOT', is_sentinel=True)
    active_nodes = deque()
    active_nodes.extend([(new_root, child) for child in root.children])

    while len(active_nodes) > 0:
        parent, node = active_nodes.pop()

        if node.is_sentinel or node_labels[node] != '':
            new_node = Node('LEAF' if node.is_sentinel else node_labels[node], is_sentinel=node.is_sentinel)
            new_node.petition_index = node.petition_index
            parent.add_child(new_node)
        else:
            new_node = parent

        for child in reversed(node.children):
            active_nodes.append((new_node, child))

    return new_root


if __name__ == '__main__':
    root_list = []
    while len(root_list) < NUM_TRIALS:
        root = generate_propagation_tree(P2, P0)
        petitions = get_all_sequences(root)
        if len(petitions) < 20:
            continue
        make_corrupted_histories(root)
        corrupted_petitions = get_all_histories(root)
        print('Max len', max(map(len, petitions)))
        print('SUCCESS', len(petitions))
        root_list.append((root, petitions, corrupted_petitions))
    sys.setrecursionlimit(10000)
    pickle.dump(root_list, open('branching_process.pickle', 'wb'))
