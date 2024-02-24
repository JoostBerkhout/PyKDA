[![PyPI version](https://badge.fury.io/py/pykda.svg)](https://badge.fury.io/py/pykda)
[![ALNS](https://github.com/JoostBerkhout/PyKDA/actions/workflows/PyKDA.yml/badge.svg)](https://github.com/JoostBerkhout/PyKDA/actions/workflows/PyKDA.yml)
[![codecov](https://codecov.io/gh/JoostBerkhout/PyKDA/graph/badge.svg?token=M4WF9A5ZML)](https://codecov.io/gh/JoostBerkhout/PyKDA)

# PyKDA

`pykda` is a Python package for the Kemeny Decomposition Algorithm (KDA) which 
allows to decompose a Markov chain into clusters of states, where states within
a cluster are relatively more connected to each other compared to states outside
the cluster. KDA was developed in the paper from [Berkhout and Heidergott (2019)](https://pubsonline-informs-org.vu-nl.idm.oclc.org/doi/10.1287/opre.2018.1813)
and uses the Kemeny constant as a connectivity measure. The Kemeny constant
is equal to the expected number of steps it takes to go from an arbitrary state
to a random state drawn according to the stationary distribution of the Markov
chain.

The package also contains Markov chain tooling for the calculations, for more
details about these calculations and the theory, please refer to the book from
[Kemeny and Snell (1976)](https://link-springer-com.vu-nl.idm.oclc.org/book/9780387901923).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install PyKDA.

```bash
pip install pykda
```

## Usage

```python
from pykda.Markov_chain import MarkovChain
import numpy as np

P = np.array(
    [
        [0.0, 0.52493781, 0.47506219],
        [0.0, 0.66666667, 0.33333333],
        [0.06666667, 0.33333333, 0.6],
        ]
    )
MC = MarkovChain(P)

print(MC.stationary_distribution.T)
print(MC.ergodic_projector)
print(MC.mean_first_passage_matrix)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change. 
Please make sure to update tests as appropriate.

## How to cite `pykda`?

If you use `pykda` in your research, please consider citing the following paper:

> Joost Berkhout, Bernd F. Heidergott (2019).
> Analysis of Markov influence graphs. 
> _Operations Research_, 67(3):892-904.
> https://doi.org/10.1287/opre.2018.1813

Or, using the following BibTeX entry:

```bibtex
@article{Berkhout2016a,
	title = {Analysis of {Markov} influence graphs},
	volume = {67},
	number = {3},
	journal = {Operations Research},
	author = {Berkhout, J. and Heidergott, B. F.},
	year = {2019},
	pages = {892--904},
}
```
