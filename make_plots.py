import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np

from reconstruct_experiments import ALPHAS

plt.rc('font', family='serif')


def plot_TED():
    """
    Generates Figure 5c
    """
    with open(f'{DATA_DIR}/errorful_distances.pickle', 'rb') as f:
        errorful = pickle.load(f)

    with open(f'{DATA_DIR}/tree_distances.pickle', 'rb') as f:
        data = pickle.load(f)[0, 0, :]

    with open(f'{DATA_DIR}/pnas_tree_distances.pickle', 'rb') as f:
        pnas_data = pickle.load(f)[0, 0, :]

    data /= 500
    pnas_data /= 500

    plt.figure(figsize=(4.5, 2.3))

    plt.plot(ALPHAS, data, label='BuildTree')
    plt.plot(ALPHAS, pnas_data, ls='dashed', label='Liben-Nowell & Kleinberg (2008)')

    plt.axhline(np.mean(errorful), ls='dashdot', c='grey', label='$\\tilde{T}$')
    plt.xlim(min(ALPHAS), max(ALPHAS))
    plt.ylim(bottom=0)
    plt.xlabel('Reconstruction Parameters ($\\alpha, \\beta$)')
    plt.ylabel('TED from True Tree')
    plt.legend()
    plt.savefig(f'zss_comparison.pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close()


def plot_err_T():
    """
    Generates Figure 5b
    """
    with open(f'{DATA_DIR}/tree_err_alpha_10.pickle', 'rb') as f:
        data = pickle.load(f)

    alphas = set()
    errs = dict()
    pnas_errs = dict()

    for alpha, err, pnas_err in data:
        alphas.add(alpha)
        if alpha not in errs:
            errs[alpha] = []
            pnas_errs[alpha] = []

        errs[alpha].append(err)
        if not np.isnan(pnas_err):
            pnas_errs[alpha].append(pnas_err)

    alphas = sorted(alphas)

    mean_errs = np.array([np.mean(errs[alpha]) for alpha in alphas])
    stddev_errs = np.array([np.std(errs[alpha]) for alpha in alphas])

    mean_pnas_errs = np.array([np.mean(pnas_errs[alpha]) for alpha in alphas])
    stddev_pnas_errs = np.array([np.std(pnas_errs[alpha]) for alpha in alphas])

    plt.figure(figsize=(4.5, 2.3))

    plt.plot(alphas, mean_errs, label='BuildTree')
    plt.fill_between(alphas, mean_errs - stddev_errs, mean_errs + stddev_errs, alpha=0.3)

    plt.plot(alphas, mean_pnas_errs, linestyle='dashed', label='Liben-Nowell & Kleinberg (2008)')
    plt.fill_between(alphas, mean_pnas_errs - stddev_pnas_errs, mean_pnas_errs + stddev_pnas_errs, alpha=0.3)

    plt.legend(loc='upper left')

    plt.xlabel('Reconstruction Parameters ($\\alpha, \\beta$)')
    plt.ylabel('err$_{10}$(T)')
    # plt.ylim(bottom=0)

    # plt.show()
    plt.savefig(f'err_comparison.pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close()


def plot_err_T_multiple_lambda():
    """
    Generates Figure 5a
    """
    fig, axes = plt.subplots(1, 4, figsize=(10, 2), sharey=True, gridspec_kw={'wspace': 0})

    for i, err_lambda in enumerate([5, 10, 15, 20]):

        data = []
        for filename in glob.glob(f'{DATA_DIR}/*/tree_err_alpha_{err_lambda}.pickle'):

            with open(filename, 'rb') as f:
                data.append(pickle.load(f))

        alphas = set()
        errs = dict()
        pnas_errs = dict()

        for trial in data:
            for alpha, err, pnas_err in trial:
                alphas.add(alpha)
                if alpha not in errs:
                    errs[alpha] = []
                    pnas_errs[alpha] = []

                errs[alpha].append(err)
                if not np.isnan(pnas_err):
                    pnas_errs[alpha].append(pnas_err)

        alphas = sorted(alphas)

        mean_errs = np.array([np.mean(errs[alpha]) for alpha in alphas])
        stddev_errs = np.array([np.std(errs[alpha]) for alpha in alphas])

        mean_pnas_errs = np.array([np.mean(pnas_errs[alpha]) for alpha in alphas])
        stddev_pnas_errs = np.array([np.std(pnas_errs[alpha]) for alpha in alphas])
        plt.axes(axes[i])

        axes[i].plot(alphas, mean_errs, label='BuildTree')
        axes[i].fill_between(alphas, mean_errs - stddev_errs, mean_errs + stddev_errs, alpha=0.3)

        axes[i].plot(alphas, mean_pnas_errs, linestyle='dashed', label='Liben-Nowell & Kleinberg (2008)')
        axes[i].fill_between(alphas, mean_pnas_errs - stddev_pnas_errs, mean_pnas_errs + stddev_pnas_errs, alpha=0.3)

        axes[i].set_ylim(0, 450000)
        axes[i].text(0.99, 0.97, f'$\\lambda$ = {err_lambda}', transform=axes[i].transAxes, va='top', ha='right')
        axes[i].set_xlim(0, 26)

    axes[0].set_ylabel(f'err$_\\lambda$(T)')
    axes[3].legend(loc=(-0.19, 0.02), framealpha=1, fontsize=8)

    fig.text(0.5, -0.04, 'Reconstruction Parameters ($\\alpha, \\beta$)', va='center', ha='center', fontsize=plt.rcParams['axes.labelsize'])

    plt.savefig(f'100_petition_err_comparison.pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close()


if __name__ == '__main__':
    DATA_DIR = 'data/15_petitions'
    plot_TED()
    plot_err_T()

    DATA_DIR = 'data/100_petitions'
    plot_err_T_multiple_lambda()

