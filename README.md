# pyKDA

pyKDA is a Python library for the Kemeny Decomposition Algorithm (KDA).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pyKDA.

```bash
pip install pyKDA
```

## Usage

```python
import PyKDA
import numpy as np

P = np.array(
    [
        [0.0, 0.52493781, 0.47506219],
        [0.0, 0.66666667, 0.33333333],
        [0.06666667, 0.33333333, 0.6],
        ]
    )
MC = MarkovChain(P)

print(MC.stationary_distribution())
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
