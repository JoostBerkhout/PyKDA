<img src="https://github.com/JoostBerkhout/PyKDA/blob/main/docs/images/logo.JPG" width="400">

[![PyPI version](https://badge.fury.io/py/pykda.svg)](https://badge.fury.io/py/pykda)
[![ALNS](https://github.com/JoostBerkhout/PyKDA/actions/workflows/PyKDA.yml/badge.svg)](https://github.com/JoostBerkhout/PyKDA/actions/workflows/PyKDA.yml)
[![codecov](https://codecov.io/gh/JoostBerkhout/PyKDA/graph/badge.svg?token=M4WF9A5ZML)](https://codecov.io/gh/JoostBerkhout/PyKDA)

`pykda` is a Python package for the Kemeny Decomposition Algorithm (KDA) which 
allows to decompose a Markov chain into clusters of states, where states within
a cluster are relatively more connected to each other than states outside
the cluster. This is useful for analyzing influence graphs, such as social 
networks and internet networks. KDA was developed in the paper from [Berkhout and Heidergott (2019)](https://research.vu.nl/ws/portalfiles/portal/104470560/Analysis_of_Markov_influence_graphs.pdf)
and uses the Kemeny constant as a connectivity measure. 

### Installing `pykda`

Package `pykda` depends on `numpy`, `tarjan` and `pyvis`.
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install PyKDA
```bash
pip install pykda
```

### Getting started

The first step is to load a Markov chain as a `MarkovChain` object using a 
transition matrix `P`.
```python
from pykda.Markov_chain import MarkovChain

P = [[0, 0.3, 0.7, 0, 0],
     [0.7, 0.2, 0.1, 0, 0],
     [0.5, 0.25, 0.25, 0, 0],
     [0, 0, 0, 0.5, 0.5],
     [0, 0, 0, 0.75, 0.25]]  # artificial transition matrix
MC = MarkovChain(P)
```
We can study some properties of the Markov chain, such as the stationary distribution:
```python
print(MC.stationary_distribution.flatten())
```
This gives `[0.226 0.156 0.23  0.232 0.156]`. We can also plot the Markov chain:
```python
MC.plot(file_name="An artificial Markov chain")
```

<img src="https://github.com/JoostBerkhout/PyKDA/blob/main/docs/images/plot_readme_example.JPG" width="280">

Now, let us decompose the Markov chain into clusters using KDA. We start by
initializing a `KDA` object using the Markov chain and the KDA settings (such
as the number of clusters). For more details about setting choices, see the [KDA documentation](https://joostberkhout.github.io/PyKDA/references/KDA/)
or [Berkhout and Heidergott (2019)](https://research.vu.nl/ws/portalfiles/portal/104470560/Analysis_of_Markov_influence_graphs.pdf).
Here, we apply the default settings, which is to cut all edges with a negative
Kemeny constant derivative and normalizing the transition matrix afterward.
```python
kda = KDA(
    original_MC=MC, CO_A="CO_A_1(1)", CO_B="CO_B_3(0)", symmetric_cut=False
    )
```
Now, let us run the KDA algorithm and visualize the results.
```python
kda.run()
kda.plot(file_name="An artificial Markov chain after KDA_A1_1_B3_0")
```

<img src="https://github.com/JoostBerkhout/PyKDA/blob/main/docs/images/plot_readme_example_after_KDA_A1_1_B3_0.JPG" width="280">

We can study the resulting Markov chain in more detail via the current Markov chain
attribute `MC` of the `KDA` object.
```python
print(kda.MC)
```
This gives the following output:
```python
MC with 5 states.
Ergodic classes: [[2, 0], [3]].
Transient classes: [[1], [4]].
```
So KDA led to a Markov multi-chain with two ergodic classes and two transient classes.
We can also study the edges that KDA cut via the `log` attribute of the `KDA` object.
```python
print(kda.log['edges cut'])
```
This gives the following output:
```
[[None], [(4, 0), (1, 4), (2, 1), (0, 1), (3, 4)]]
```
We can also study the Markov chains that KDA found in each (outer) iteration via
 `kda.log['Markov chains']`)`.

As another KDA application example, let us apply KDA until we find two ergodic 
classes explicitly. We will also ensure that the Kemeny constant derivatives are
recalculated after each cut (and normalize the cut transition matrix to 
ensure it is a stochastic matrix again). To that end, we use:
```python
kda2 = KDA(
    original_MC=MC, CO_A="CO_A_2(2)", CO_B="CO_B_1(1)", symmetric_cut=False
    )
kda2.run()
kda2.plot(file_name="An artificial Markov chain after KDA_A2_2_B1_1")
```
which gives (edges (4, 0) and (1, 4) are cut in two iterations):

<img src="https://github.com/JoostBerkhout/PyKDA/blob/main/docs/images/plot_readme_example_after_KDA_A2_2_B1_1.JPG" width="280">

### How to learn more about `pykda`?
To learn more about `pykda` have a look at the [documentation](https://joostberkhout.github.io/PyKDA/). There, you can
also find links to interactive Google Colab notebooks in [examples](https://joostberkhout.github.io/PyKDA/examples/). If you
have any questions, feel free to open an issue here on [Github Issues](https://github.com/JoostBerkhout/PyKDA/issues).

### How to cite `pykda`?

If you use `pykda` in your research, please consider citing the following paper:

> Joost Berkhout, Bernd F. Heidergott (2019).
> Analysis of Markov influence graphs. 
> _Operations Research_, 67(3):892-904.
> https://doi.org/10.1287/opre.2018.1813

Or, using the following BibTeX entry:

```bibtex
@article{Berkhout_Heidergott_2019,
	title = {Analysis of {Markov} influence graphs},
	volume = {67},
	number = {3},
	journal = {Operations Research},
	author = {Berkhout, J. and Heidergott, B. F.},
	year = {2019},
	pages = {892--904},
}
```
