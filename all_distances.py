from glob import glob
import sys
import pickle
from multiprocessing import Pool, cpu_count
from editdistance import eval as ed
from itertools import combinations_with_replacement
from tqdm import tqdm

USAGE = 'python3 all_distances.py IN_DIR [THREADS]'


def get_pair_distances(sequence_pair):
    """
    Computer all pairwise edit distances between strings in a pair of sequences
    :param sequence_pair: a tuple of two string sequences
    :return: a dictionary mapping string tuples (x, y) to ED(x, y)
    """
    pair_distances = dict()
    for string1 in sequence_pair[0]:
        for string2 in sequence_pair[1]:
            pair_distances[string1, string2] = ed(string1, string2)
            pair_distances[string2, string1] = pair_distances[string1, string2]
    return pair_distances


def get_distances(sequences, threads):
    """
    Compute all string edit distances within a set of string sequences, using the given number of threads.
    :param sequences: a list of string sequences
    :param threads: an int
    :return: a dictionary mapping string tuples (x, y) to ED(x, y)
    """
    sequences.append([''])  # Make sure distance to empty is computed
    pool = Pool(threads)
    num_sequence_pairs = len(sequences) * (len(sequences) + 1) // 2
    distances = dict()

    for result in tqdm(pool.imap_unordered(get_pair_distances, combinations_with_replacement(sequences, 2)),
                       total=num_sequence_pairs):
        distances.update(result)

    pool.close()
    pool.join()

    sequences.remove([''])
    return distances


def get_distances_sequential(sequences):
    """
    Compute all string edit distances within a set of string sequences, using just the main thread and with no progress
    bar. Used for running experiments.
    :param sequences: a list of string sequences
    :return: a dictionary mapping string tuples (x, y) to ED(x, y)
    """
    sequences.append([''])  # Make sure distance to empty is computed
    distances = dict()

    for seq1, seq2 in combinations_with_replacement(sequences, 2):
        for string1 in seq1:
            for string2 in seq2:
                distances[string1, string2] = ed(string1, string2)
                distances[string2, string1] = distances[string1, string2]

    sequences.remove([''])
    return distances


if __name__ == '__main__':
    try:
        in_dir = sys.argv[1]
        threads = cpu_count() // 2
        if len(sys.argv) > 2:
            threads = int(sys.argv[2])
    except Exception as e:
        print('Error:', e)
        print('Usage:', USAGE)
        exit(1)

    sequences = []
    for file in glob(in_dir + '/*.txt'):
        with open(file) as f:
            sequences.append(f.read().strip().split('\n'))

    print('Computing all string EDs on {} threads...'.format(threads))
    distances = get_distances(sequences, threads)

    with open(in_dir + '/distances.pickle', 'wb') as f:
        pickle.dump(distances, f)

    print('Saved pickle to {}/distances.pickle'.format(in_dir))
