import functools
from collections import deque
import networkx as nx


import numpy as np


class Node:
    traverse_index = 0

    # This specifies what instance variables this class uses and improves storage efficiency when pickling in
    # conjunction with __getstate__ and __setstate__.
    __slots__ = 'label', 'children', 'parent', 'history', 'leaves', 'petition_index', 'is_sentinel'

    def __init__(self, label, petition_index=None, is_sentinel=False):
        self.label = label
        self.children = []
        self.parent = None
        self.history = []
        self.leaves = []
        self.petition_index = petition_index
        self.is_sentinel = is_sentinel

    def clear_history(self):
        """
        Recursively empty the history lists in this tree.
        """
        self.history = []
        for child in self.children:
            child.clear_history()

    def label_by_medoid(self, distances):
        """
        Recursively set the labels of this tree to be the medoids of the node label lists, using precomputed EDs.
        :param distances: a dicts from keys (a, b) to ED(a, b)
        """
        if not isinstance(self.label, str):
            self.label = find_medoid(self.label, distances)
        for child in self.children:
            child.label_by_medoid(distances)

    def dot(self, dot, parent_id=None, include_empty=True):
        """
        Recursively construct a graphviz representation of this tree, using calls to .node() and .edge()
        :param dot: a graphviz object
        :param parent_id: the unique id of the current parent node (None if we're the first call, ie root)
        :param include_empty: whether to include nodes with empty labels
        """
        me = Node.traverse_index
        draw_me = include_empty or self.label != ''

        if draw_me:
            dot.node(str(me), str(self.label))

            if parent_id is not None:
                dot.edge(str(parent_id), str(me))

        Node.traverse_index += 1
        for child in self.children:
            child.dot(dot, parent_id=me if draw_me else parent_id, include_empty=include_empty)

    def add_child(self, *nodes):
        """
        Add children Nodes to this Node.
        :param nodes: one or more Nodes to add as children
        """
        for node in nodes:
            self.children.append(node)
            node.parent = self

    def remove_child(self, *nodes):
        """
        Remove children from this node.
        :param nodes: one or more Nodes that are currently children of self
        """
        for node in nodes:
            self.children.remove(node)

    def __repr__(self):
        """
        The string representation of this Node used by print().
        :return: the string
        """
        return 'Node({})'.format(self.label)

    def sort_by_label(self):
        """
        Recursively order children in this tree alphabetically by label.
        """
        self.children.sort(key=lambda x: x.label)
        for child in self.children:
            child.sort_by_label()

    def compute_subtree_leaves(self):
        """
        Find the leaves under every subtree and store them in self.leaves, so that we can sort children by inversions.
        :return: the leaves under self
        """
        if len(self.children) == 0:
            self.leaves = [self]
            # assert self.label == 'LEAF'
            assert self.petition_index is not None
        else:
            self.leaves = []
            for child in self.children:
                self.leaves.extend(child.compute_subtree_leaves())

        return self.leaves

    def sort_by_inversions(self):
        """
        Recursively sort children in this tree by petition index inversions. If necessary, first compute the leaves of
        every subtree.
        """
        if len(self.leaves) == 0:
            self.compute_subtree_leaves()

        self.children.sort(key=inversion_cmp)
        for child in self.children:
            child.sort_by_inversions()

    def __getstate__(self):
        """
        Make a dict storing the value of each instance variable. Improves storage efficiency when pickling.
        :return: the dict
        """
        return dict(
            (slot, getattr(self, slot))
            for slot in self.__slots__
            if hasattr(self, slot)
        )

    def __setstate__(self, state):
        """
        Set the values of instance variables according to a dict returned by __getstate__. Allows unpickling from
        objects pickled in this way.
        :param state: the dict storing the values of variables
        """
        for slot, value in state.items():
            setattr(self, slot, value)

    def to_networkx_digraph(self, G=None):
        if G is None:
            G = nx.DiGraph()
            if not self.is_sentinel:
                G.add_node(self.label)

        for child in self.children:
            if not self.is_sentinel and not child.is_sentinel:
                G.add_edge(self.label, child.label)
            child.to_networkx_digraph(G)

        return G

    def remove_empty_nodes(self, parent=None):
        if self.label == '' and parent is not None:
            for child in self.children:
                parent.add_child(child)
                child.remove_empty_nodes(parent)
            parent.children.remove(self)
        else:
            for child in self.children:
                child.remove_empty_nodes(self)

    def remove_sentinels(self):
        new_children = [child for child in self.children if not child.is_sentinel]
        self.children = new_children
        for child in self.children:
            child.remove_sentinels()

        if self.is_sentinel and len(self.children) == 1:
            self.children[0].parent = None
            return self.children[0]



@functools.cmp_to_key
def inversion_cmp(u, v):
    """
    A sort key that allows sorting children to minimize the number of petition index inversions. This method returns a
    Java compareTo()-style int, which cmp_to_key converts into a key that Python's sort() can use.
    :param u: a Node
    :param v: another Node
    :return: a positive int if u should be to the right of v, negative if the reverse, or 0 if tied
    """
    count = 0
    for u_leaf in u.leaves:
        for v_leaf in v.leaves:
            if u_leaf.petition_index < v_leaf.petition_index:
                count -= 1
            elif u_leaf.petition_index > v_leaf.petition_index:
                count += 1
    return count


def find_medoid(strings, distances):
    """
    Compute the medoid of a list of strings, with precomputed EDs. The medoid of A is the elt of A whose sum of ED to
    strings in A is minimal
    :param strings: a list of strings
    :param distances: a dict from keys (a, b) to ED(a, b)
    :return: the medoid of strings
    """
    min_sum = np.inf
    medoid = ''

    for string in strings:
        total = sum(distances[string, other] for other in strings)

        if total < min_sum:
            min_sum = total
            medoid = string

    return medoid


def get_label_sequence(leaf):
    """
    Find the root-leaf label sequence of a leaf.
    :param leaf: a Node
    :return: a list of string labels
    """
    label_sequence = []
    current = leaf
    while current is not None:
        if not current.is_sentinel:
            label_sequence.insert(0, current.label)
        current = current.parent
    return label_sequence


def get_all_sequences(root):
    """
    Get all root-leaf label sequences in a tree
    :param root: a Node
    :return: a list of lists of strings
    """
    petitions = []
    leaves = get_leaves(root)
    for leaf in leaves:
        petitions.append(get_label_sequence(leaf))
    return petitions


def get_leaves(root):
    """
    Find the leaves of a tree, in petition index order (if defined), else in left-right order
    :param root: a Node
    :return: a list of Nodes
    """
    leaves = []
    active_nodes = deque()
    active_nodes.append(root)

    while len(active_nodes) > 0:
        current = active_nodes.pop()
        if len(current.children) == 0:
            leaves.append(current)
        else:
            for i in range(len(current.children)-1, -1, -1):
                active_nodes.append(current.children[i])

    if leaves[0].petition_index is not None:
        leaves.sort(key=lambda x: x.petition_index)

    return leaves


def get_all_histories(root):
    """
    Get the histories (corrupted sequences) from a tree (ignoring sentinel nodes and empty--deleted--labels)
    :param root: a Node
    :return: a list of lists of strings
    """
    leaves = get_leaves(root)
    histories = []
    for leaf in leaves:
        if leaf.is_sentinel:
            leaf = leaf.parent
        histories.append([label for label, node in leaf.history if label != ''])
    return histories


def count_nodes(root):
    """
    Count the number of (non-sentinel) nodes in a tree
    :param root: a Node
    :return: the int number of nodes
    """
    count = 0
    active_nodes = deque()
    active_nodes.append(root)

    while len(active_nodes) > 0:
        current = active_nodes.pop()
        active_nodes.extend(current.children)
        if not current.is_sentinel:
            count += 1

    return count


def divergence_depth(u, v):
    """
    Find the depth of the LCA of u and v.
    :param u: a Node
    :param v: another Node in the same tree as u
    :return: the int depth of LCA(u, v)
    """

    current = u
    while v not in current.leaves:
        current = current.parent

    depth = 0
    while not current.is_sentinel:
        current = current.parent
        depth += 1

    return depth
