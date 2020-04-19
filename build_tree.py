from glob import glob
from itertools import combinations

import zss
from graphviz import Digraph
from tqdm import tqdm

from all_distances import get_distances
from petitions_branching_process import *
from pnas_build_tree import pnas_build_tree
from tree_error import tree_error

INF = np.inf
THREADS = 6
ALPHA = 15

START = (0, 0)
GIVEUP = None
INSERT = (0, 1)
DELETE = (1, 0)
SUBSTITUTE = (1, 1)


def internal_disagreement(string_list, distances):
    """
    Compute the sum of the edit distances between the medoid of string_list and each string in string_list.
    :param string_list: a list of strings
    :param distances: a dictionary mapping tuples (x, y) to ED(x, y)
    :return: the edit distance sum
    """
    min_ed_sum = INF
    for string in string_list:
        ed_sum = 0
        for other in string_list:
            ed_sum += distances[string, other]
        if ed_sum < min_ed_sum:
            min_ed_sum = ed_sum

    return min_ed_sum


def cost(pair1, pair2, distances):
    """
    Compute the cost of substituting one string list for another. The string lists come with precomputed internal
    disagreement values.
    :param pair1: a (string_list1, internal_disagreement1) pair
    :param pair2: a (string_list2, internal_disagreement2) pair
    :param distances: a dictionary mapping tuples (x, y) to ED(x, y)
    :return: the cost of substituting list1 for list 2
    """
    list1, internal1 = pair1
    list2, internal2 = pair2

    return internal_disagreement(list1 + list2, distances) - internal1 - internal2


def divergent_align(x, y, alpha, distances):
    """
    Align the sequences of list of strings a and b, with parameter alpha and precomputed edit distances. The sequences
    need to also include precomputed internal disagreement values.
    :param x: a sequence of tuples (string_list, internal_disagreement)
    :param y: a sequence of tuples (string_list, internal_disagreement)
    :param alpha: the node cost parameter
    :param distances: a dictionary mapping tuples (x, y) to ED(x, y)
    :return: the prefix, two branches of the alignment, and substitution count, as a tuple
    """
    # A list of empty strings for computing delete costs
    delete_empties = []
    if len(y) > 0:
        delete_empties = [''] * len(y[0][0])

    # A list of empty strings for computing insert costs
    insert_empties = []
    if len(x) > 0:
        insert_empties = [''] * len(x[0][0])

    # dist[i][j] is a tuple (d, back_pointer), d = the Levenshtein distance between x[i:] and y[j:]
    dist = [[tuple() for _ in range(len(y) + 1)] for _ in range(len(x) + 1)]
    # Initialize matrix
    dist[len(x)][len(y)] = (0, START)

    # Fill in rightmost column
    # a[i:] -> '' = deleting each character in a[i:]
    for i in range(len(x) - 1, -1, -1):
        delete = dist[i+1][len(y)][0] + cost(x[i], (delete_empties, 0), distances) + alpha
        giveup = alpha * (len(x) - i)
        dist[i][len(y)] = (delete, DELETE) if delete < giveup else (giveup, GIVEUP)

    # Fill in bottommost row
    # '' -> b[j:] = inserting each character in b[j:]
    for j in range(len(y) - 1, -1, -1):
        insert = dist[len(x)][j + 1][0] + cost((insert_empties, 0), y[j], distances) + alpha
        giveup = alpha * (len(y) - j)
        dist[len(x)][j] = (insert, INSERT) if insert < giveup else (giveup, GIVEUP)

    # Fill in the rest of the dist matrix right to left, bottom to top
    for i in range(len(x) - 1, -1, -1):
        for j in range(len(y) - 1, -1, -1):
            delete = dist[i+1][j][0] + cost(x[i], (delete_empties, 0), distances) + alpha
            insert = dist[i][j+1][0] + cost((insert_empties, 0), y[j], distances) + alpha
            sub = dist[i+1][j+1][0] + cost(x[i], y[j], distances) + alpha
            giveup = alpha * (len(x) - i + len(y) - j)

            if sub < giveup and sub < insert and sub < delete:
                dist[i][j] = (sub, SUBSTITUTE)
            elif giveup < insert and giveup < delete:
                dist[i][j] = (giveup, GIVEUP)
            elif delete < insert:
                dist[i][j] = (delete, DELETE)
            else:
                dist[i][j] = (insert, INSERT)
                   
    # Build merged prefix (with new internal disagreements) and make branches
    row, col = 0, 0
    back_pointer = dist[row][col][1]
    prefix = []
    x_branch = []
    y_branch = []
    substitution_count = 0

    while back_pointer != START:
        if back_pointer == DELETE:
            prefix.append((x[row][0] + delete_empties, internal_disagreement(x[row][0] + delete_empties, distances)))
        elif back_pointer == INSERT:
            prefix.append((insert_empties + y[col][0], internal_disagreement(insert_empties + y[col][0], distances)))
        elif back_pointer == SUBSTITUTE:
            prefix.append((x[row][0] + y[col][0], internal_disagreement(x[row][0] + y[col][0], distances)))
            substitution_count += 1
        elif back_pointer == GIVEUP:
            while row < len(x):
                x_branch.append(x[row])
                row += 1
            while col < len(y):
                y_branch.append(y[col])
                col += 1
            break

        row, col = row + back_pointer[0], col + back_pointer[1]
        back_pointer = dist[row][col][1]

    return prefix, x_branch, y_branch, substitution_count, dist[0][0][0]


def build_tree_from_files(in_dir, alpha):
    """
    Given a directory with sequences in .txt files and a distance.pickle file, build the tree according to alpha.
    :param in_dir: the directory in which the sequences and distance file are stored
    :param alpha: the node cost parameter
    """
    sequences = []
    for file in glob(in_dir + '/*.txt'):
        with open(file) as f:
            sequences.append(f.read().strip().split('\n'))

    with open(in_dir + '/distances.pickle', 'rb') as f:
        distances = pickle.load(f)

    return build_tree(sequences, alpha, distances)


def build_tree(sequences, alpha, distances, verbose=True):
    """
    Build a tree from a list of string sequences.
    :param sequences: a list of string sequences
    :param alpha: the node cost parameter
    :param distances: a dict of string tuples (x, y) to ED(x, y)
    :return: the Node root of the reconstructed tree
    """
    sequences = sequences[:]
    # wrap every name in list, put that in a tuple with the internal disagreement
    for i, p in enumerate(sequences):
        sequences[i] = [([x], 0) for x in p]

    active_indices = list(range(len(sequences)))
    sequence_alignments = dict()

    pair_iter = combinations(active_indices, 2)
    if verbose:
        pair_iter = tqdm(pair_iter, total=len(active_indices) * (len(active_indices) - 1) // 2)
        print('Aligning all sequences...')

    for i, j in pair_iter:
        sequence_alignments[i, j] = divergent_align(sequences[i], sequences[j], alpha, distances)

    branches = dict()

    for _ in tqdm(range(len(active_indices) - 1)) if verbose else range(len(active_indices) - 1):
        # Find pair with most substitutions in shared trunk
        to_merge1, to_merge2 = max(((i, j) for i in active_indices for j in active_indices if i < j),
                                   key=lambda pair: sequence_alignments[pair][3])

        trunk, branch1, branch2, _, _ = sequence_alignments[to_merge1, to_merge2]

        # Add trunk to sequences, remove merged sequences
        sequences.append(trunk)
        new = len(sequences) - 1
        active_indices.remove(to_merge1)
        active_indices.remove(to_merge2)
        
        # Realign all to new petition
        for old in active_indices:
            sequence_alignments[old, new] = divergent_align(sequences[old], sequences[new], alpha, distances)

        active_indices.append(new)
        branches[new] = []

        # Make nodes for the two branches
        for petition_index, branch in ((to_merge1, branch1), (to_merge2, branch2)):
            if len(branch) > 0:
                branch_root = Node(branch[0][0])
                current = branch_root
                for index in range(1, len(branch)):
                    node = Node(branch[index][0])
                    current.add_child(node)
                    current = node

                # If there are branches off this branch, connect them
                if petition_index in branches:
                    for node in branches[petition_index]:
                        current.add_child(node)
                else:
                    # Otherwise, make dummy leaves
                    dummy = Node('LEAF', petition_index=petition_index, is_sentinel=True)
                    current.add_child(dummy)

                branches[new].append(branch_root)
            else:
                # If this branch has length 0, just add its branches or make a leaf for it
                if petition_index in branches:
                    branches[new].extend(branches[petition_index])
                else:
                    branches[new].append(Node('LEAF', petition_index=petition_index, is_sentinel=True))

    # Now, there's only one sequence left: the main trunk of the tree
    final_index = active_indices.pop()
    assert len(active_indices) == 0
    trunk = sequences[final_index]

    # Make nodes for trunk, with a dummy root
    root = Node('ROOT', is_sentinel=True)
    current = root
    for entry in trunk:
        node = Node(entry[0])
        current.add_child(node)
        current = node

    # Add branches off trunk
    if final_index in branches:
        for node in branches[final_index]:
            current.add_child(node)

    root.label_by_medoid(distances)
    return root


if __name__ == '__main__':
    sys.setrecursionlimit(3000)

    random.seed(1)
    np.random.seed(1)

    print('Branching process...')
    done = False
    while not done:
        root = generate_propagation_tree(P2, P0)
        sequences = get_all_sequences(root)
        print(len(sequences))
        done = 100 <= len(sequences) < 101

    print('Done. Mean length:', np.mean([len(seq) for seq in sequences]))

    dot = Digraph()
    root.dot(dot)
    dot.render('real.gv', view=True)

    make_corrupted_histories(root, 0.001, 0.001, 0.05, 0.05)
    corrupted_sequences = get_all_histories(root)

    print('Computing all edit distances...')
    distances = get_distances(corrupted_sequences + sequences, THREADS)

    reconstructed_root = build_tree(corrupted_sequences[:], ALPHA, distances)

    print('err(T) =', tree_error(reconstructed_root, corrupted_sequences, ALPHA, distances))

    print('\nReconstructed tree to truth:')
    # print('Computing unordered dist....')
    print('unordered dist', zss.simple_distance(root, reconstructed_root))

    print('Sorting by inversions...')
    reconstructed_root.sort_by_inversions()
    reconstructed_root = reconstructed_root.remove_sentinels()
    # reconstructed_root.remove_empty_nodes()

    dot = Digraph()
    reconstructed_root.dot(dot)
    dot.render('ordered_reconstructed.gv', view=True)
    print('ordered dist', zss.simple_distance(root, reconstructed_root))

    dot = Digraph()
    reconstructed_root.dot(dot, include_empty=False)
    dot.render('no_blank_ordered_reconstructed.gv', view=True)

    errorful_root = get_corrupted_tree(root, distances)
    dot = Digraph()
    errorful_root.dot(dot)
    dot.render('errorful.gv', view=True)

    print('err(T\') =', tree_error(errorful_root, corrupted_sequences, ALPHA, distances))

    print('\nErrorful tree to truth:')
    print('unordered dist', zss.simple_distance(root, errorful_root))

    print('\nErrorful tree to reconstructed:')
    print('unordered dist', zss.simple_distance(reconstructed_root, errorful_root))

    reconstructed_root = pnas_build_tree(corrupted_sequences[:], 5)
    print('pnas dist', zss.simple_distance(root, reconstructed_root))
    print('err(T) =', tree_error(reconstructed_root, corrupted_sequences, ALPHA, distances))




