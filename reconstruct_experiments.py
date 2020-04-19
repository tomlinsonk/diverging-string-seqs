import os
from functools import partial
from multiprocessing import Pool

import zss
from tqdm import tqdm

from all_distances import get_distances_sequential
from build_tree import build_tree
from petitions_branching_process import *
from pnas_build_tree import pnas_build_tree
from tree_error import tree_error


NUM_THREADS = 8

WORD_ERRORS = [0.001]
CHAR_ERRORS = [0.1]

ALPHAS = list(range(1, 26))
PNAS_ED_THRESHOLDS = list(range(1, 26))


def reconstruct_pnas(args):
    """
    Reconstruct trees using the PNAS method.
    Helper method to be run in a multiprocessing pool.
    :param args: the tuple (sequences, pnas_ed_threshold)
    :return: the tuple with the results
    """
    word_err, char_err, sequences, pnas_ed_threshold = args

    reconstructed_root = pnas_build_tree(sequences, pnas_ed_threshold)

    return word_err, char_err, pnas_ed_threshold, reconstructed_root


def corrupt_and_reconstruct(args):
    """
    Generate corrupted sequences from a tree and use build_tree on the resulting sequences.
    Helper method to be run in a multiprocessing pool.
    :param args: a tuple containing (word_error_rate, char_error_rate, alpha, trial_index, root)
    :return: the tuple with the results
    """
    word_err, char_err, alpha, _, root = args
    root.clear_history()
    make_corrupted_histories(root, word_sub=word_err, word_del=word_err, char_sub=char_err, char_del=char_err)
    corrupted_sequences = get_all_histories(root)

    distances = get_distances_sequential(corrupted_sequences)
    reconstructed_root = build_tree(corrupted_sequences, alpha, distances, verbose=False)
    reconstructed_root.sort_by_inversions()

    return word_err, char_err, alpha, reconstructed_root, corrupted_sequences


def corrupt_and_errorful_distance(args):
    """
    Generate a corrupted tree (correct topology but error in labels) and compute its zss distance to the real tree.
    Helper method to be run in a multiprocessing pool.
    :param args: a tuple with error rates and the real tree
    :return:
    """
    word_err, char_err, _, root = args

    root.clear_history()
    make_corrupted_histories(root, word_sub=word_err, word_del=word_err, char_sub=char_err, char_del=char_err)
    corrupted_sequences = get_all_histories(root)
    distances = get_distances_sequential(corrupted_sequences)

    return zss.simple_distance(root, get_corrupted_tree(root, distances))


def compute_tree_distances(trial, true_root):
    """
    Compute the zss distance between the real tree and a reconstructed one. Helper method to be run in a multiprocessing
    pool.
    :param trial: the tuple including the reconstructed tree.
    :param true_root: the real tree
    :return: a tuple including the distance and params
    """
    word_err, char_err, alpha, reconstructed_root, corrupted_sequences = trial
    distance = zss.simple_distance(true_root, reconstructed_root)

    return (word_err, char_err, alpha), distance


def compute_pnas_tree_distances(trial, true_root):
    """
    Compute the zss distance between the real tree and a reconstructed one. Helper method to be run in a multiprocessing
    pool.
    :param trial: the tuple including the reconstructed tree.
    :param true_root: the real tree
    :return: a tuple including the distance and params
    """
    word_err, char_err, pnas_ed_threshold, reconstructed_root = trial
    if reconstructed_root is None:
        return (word_err, char_err, pnas_ed_threshold), np.nan

    distance = zss.simple_distance(true_root, reconstructed_root)

    return (word_err, char_err, pnas_ed_threshold), distance


def compute_err_helper(args, err_alpha):
    word_err, char_err, alpha, reconstructed_root, corrupted_sequences = args
    distances = get_distances_sequential(corrupted_sequences)

    pnas_root = pnas_build_tree(corrupted_sequences, alpha)

    return alpha, tree_error(reconstructed_root, corrupted_sequences, err_alpha, distances), tree_error(pnas_root, corrupted_sequences, err_alpha, distances)


def reconstruct_experiment():
    """
    Generate a branching process tree, and run corruptions and reconstructions in parallel. Save the results in a pickle
    file.
    """
    print('Running branching process...')
    done = False
    while not done:
        root = generate_propagation_tree(P2, P0)
        done = MIN_SEQUENCES <= len(get_leaves(root)) <= MAX_SEQUENCES

    param_vals = [(w_err, c_err, alpha, trial, root)
                  for w_err in WORD_ERRORS
                  for c_err in CHAR_ERRORS
                  for alpha in ALPHAS
                  for trial in range(TRIALS)]

    print('Running reconstruct_experiment...')
    with Pool(NUM_THREADS) as pool:
        reconstructions = list(tqdm(pool.imap_unordered(corrupt_and_reconstruct, param_vals), total=len(param_vals)))

    with open(f'{DATA_PATH}/tree_and_reconstructions.pickle', 'wb') as f:
        pickle.dump((root, reconstructions), f)


def run_pnas_on_corrupted_sequences():
    """
    In parallel, use the PNAS method on data in a reconstructed tree file and compute the distances to the real tree.
    """
    with open(f'{DATA_PATH}/tree_and_reconstructions.pickle', 'rb') as f:
        true_root, reconstructions = pickle.load(f)

    param_vals = [(word_err, char_err, corrupted_sequences, pnas_ed_threshold)
                  for word_err, char_err, alpha, _, corrupted_sequences in reconstructions
                  for pnas_ed_threshold in PNAS_ED_THRESHOLDS if alpha == pnas_ed_threshold]

    print('Running run_pnas_on_corrupted_sequences (reconstruction phase)...')
    with Pool(NUM_THREADS) as pool:
        pnas_reconstructions = list(tqdm(pool.imap_unordered(reconstruct_pnas, param_vals), total=len(param_vals)))

    with open(f'{DATA_PATH}/pnas_reconstructions.pickle', 'wb') as f:
        pickle.dump(pnas_reconstructions, f)

    data = np.zeros((len(WORD_ERRORS), len(CHAR_ERRORS), len(PNAS_ED_THRESHOLDS)))
    compute_dist = partial(compute_pnas_tree_distances, true_root=true_root)
    print('Running run_pnas_on_corrupted_sequences (compute distances phase)...')
    with Pool(NUM_THREADS) as pool:
        for params, distance in tqdm(pool.imap_unordered(compute_dist, pnas_reconstructions), total=len(pnas_reconstructions)):
            word_err, char_err, pnas_ed_threshold = params
            data[WORD_ERRORS.index(word_err), CHAR_ERRORS.index(char_err), PNAS_ED_THRESHOLDS.index(pnas_ed_threshold)] += distance

    with open(f'{DATA_PATH}/pnas_tree_distances.pickle', 'wb') as f:
        pickle.dump(data, f)


def compute_all_tree_distances():
    """
    In parallel, compute zss distances between reconstructed trees loaded from a pickle and the real tree in that
    pickle.
    """
    data = np.zeros((len(WORD_ERRORS), len(CHAR_ERRORS), len(ALPHAS)))
    with open(f'{DATA_PATH}/tree_and_reconstructions.pickle', 'rb') as f:
        true_root, reconstructions = pickle.load(f)

    compute_dist = partial(compute_tree_distances, true_root=true_root)

    print('Running compute_all_tree_distances...')
    with Pool(NUM_THREADS) as pool:
        for params, distance in tqdm(pool.imap_unordered(compute_dist, reconstructions), total=len(reconstructions)):
            word_err, char_err, alpha = params
            data[WORD_ERRORS.index(word_err), CHAR_ERRORS.index(char_err), ALPHAS.index(alpha)] += distance

    with open(f'{DATA_PATH}/tree_distances.pickle', 'wb') as f:
        pickle.dump(data, f)


def compute_errorful_distance():
    """
    In parallel, generate some errorful trees and compute their zss distance to the real tree. Save the distances in a
    pickle.
    """
    with open(f'{DATA_PATH}/tree_and_reconstructions.pickle', 'rb') as f:
        true_root, _ = pickle.load(f)

    param_vals = [(w_err, c_err, trial, true_root)
                  for w_err in WORD_ERRORS
                  for c_err in CHAR_ERRORS
                  for trial in range(TRIALS)]

    print('Running compute_errorful_distance...')
    with Pool(NUM_THREADS) as pool:
        dists = list(tqdm(pool.imap_unordered(corrupt_and_errorful_distance, param_vals), total=len(param_vals)))

    with open(f'{DATA_PATH}/errorful_distances.pickle', 'wb') as f:
        pickle.dump(dists, f)


def compute_err_T(err_alpha):
    """
    In parallel, compute err(T) for the given alpha
    :return:
    """

    data = []

    with open(f'{DATA_PATH}/tree_and_reconstructions.pickle', 'rb') as f:
        true_root, reconstructions = pickle.load(f)

    helper_partial = partial(compute_err_helper, err_alpha=err_alpha)

    print(f'Running compute err_{err_alpha}(T)...')
    with Pool(NUM_THREADS) as pool:
        for alpha, err, pnas_err in tqdm(pool.imap_unordered(helper_partial, reconstructions), total=len(reconstructions)):
            data.append((alpha, err, pnas_err))

    with open(f'{DATA_PATH}/tree_err_alpha_{err_alpha}.pickle', 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    sys.setrecursionlimit(10000)

    # Run 100 petition experiment
    TRIALS = 1
    MIN_SEQUENCES = 100
    MAX_SEQUENCES = 100

    for run in range(8):
        print('RUN', run)
        random.seed(run)
        np.random.seed(run)

        DATA_PATH = f'data/100_petitions/run_{run}'
        os.makedirs(DATA_PATH, exist_ok=True)

        reconstruct_experiment()

        # These must be run after reconstruct_experiment(), as they rely on the output pickle.
        run_pnas_on_corrupted_sequences()
        for err_alpha in [5, 10, 15, 20]:
            compute_err_T(err_alpha)

    # Run 15 petition experiment
    TRIALS = 500
    MIN_SEQUENCES = 15
    MAX_SEQUENCES = 15

    DATA_PATH = f'data/15_petitions'
    os.makedirs(DATA_PATH, exist_ok=True)

    reconstruct_experiment()

    # These must be run after reconstruct_experiment(), as they rely on the output pickle.
    run_pnas_on_corrupted_sequences()
    compute_all_tree_distances()
    compute_errorful_distance()
    compute_err_T(10)
