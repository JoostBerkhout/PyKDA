# PyKDA

Welcome to the Python package `pykda`. It implements the Kemeny Decomposition 
Algorithm (KDA) from [Berkhout and Heidergott (2019)](https://research.vu.nl/ws/portalfiles/portal/104470560/Analysis_of_Markov_influence_graphs.pdf)
 which allows to decompose a Markov chain into clusters of states, 
where states within a cluster are relatively more connected to each other compared
to states outside the cluster. 

KDA uses the Kemeny constant as a connectivity measure.
The Kemeny constant is equal to the expected number of steps it takes to go from
an arbitrary state to a random state drawn according to the stationary distribution
of the Markov chain.

### About

This package is an attempt to make the code from [Berkhout and Heidergott (2019)](https://research.vu.nl/ws/portalfiles/portal/104470560/Analysis_of_Markov_influence_graphs.pdf)
more accessible to the public. Also, it serves as a practice project for the
author (Joost Berkhout) to learn about software package development in Python.

`pykda` uses Python 3.10+ and depends on the following packages:

- `numpy`: for numerical linear algebra calculations.
- `tarjan`: to determine strongly connected components in graphs.
- `pyvis`: to visualize the Markov chain and the KDA clusters.

### Acknowledgement
The author would like to thank [Leon Lan](https://github.com/leonlan) for his help
and inspiration. The setup in this package tries to follow that of the package [ALNS](https://github.com/N-Wouda/ALNS).
