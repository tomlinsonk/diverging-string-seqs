# Summarizing Diverging String Sequences, with Applications to Chain-Letter Petitions
This repository contains code and data to accompany the paper
> Patty Commins, David Liben-Nowell, Tina Liu, and Kiran Tomlinson. Summarizing Diverging String Sequences, with Applications to Chain-Letter Petitions. 31st Annual Symposium on Combinatorial Pattern Matching (CPM), June 2020.

## Contents
- `all_distances.py`: for precomputing edit distance between all pairs of strings across two sequences
- `build_tree.py`: contains the main reconstruction algorithm presented in the paper and supporting methods
- `make_plots.py`: for making the three plots in the paper
- `node.py`: supporting code for storing and manipulating trees
- `petitions_branching_process.py`: for simulating data
- `pnas_build_tree.py`: an implementation of the reconstruction method from 
  > Liben-Nowell, D., & Kleinberg, J. (2008). Tracing information flow on a global scale using Internet chain-letter data. Proceedings of the National Academy of Sciences, 105(12), 4633-4638.
- `reconstruct_experiments.py`: for running our experiments
- `tree_error.py`: methods for computing err(T) and AED(x, y)
- `data/`
    - `100_petitions/`: results from the eight trials used in the *m* = 100 experiment (used to generate Figure 5a)
    - `15_petitions/`: results from the *m* = 15 experiment (used to generate Figure 5bc)

## Libraries
We used Python 3.6.8 (also tested with Python 3.7.6) along with the following libraries:
- `editdistance 0.5.3`
- `zss 1.2.0`
- `tqdm 4.32.2`
- `graphviz 0.11 `
- `matplotlib 3.2.1`
- `numpy 1.18.1`
- `networkx 2.4`

## Reproducibility
To run both the 15- and 100-petition experiments, just run `python3 reconstruct_experiments.py`.
The results will be placed in `data/`. The results from our runs are provided in the repo. 
We have not included the files `pnas_reconstructions.pickle` and `tree_and_reconstructions.pickle`
from each experiment, since they are very large (if you really want them, send me an email or open an
issue and I'll be happy to share them). The eight trials of the 100-petition experiment have their RNGs
seeded with the numbers 0-8, so you should get exactly the same results (modulo things out of our control, like
OS/Python updates). We didn't seed the 15-petition experiment, but 500 trials is plenty to even out random
variance. 

To generate the three plots in the paper, run `python3 make_plots.py`. This uses the files stored in `data/`.
Using the data files provided here will generate the plots exactly as they appear in the paper. 