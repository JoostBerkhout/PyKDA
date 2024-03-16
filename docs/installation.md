# Installation instructions

## Installation

The package can be installed through pip by running the following command in the terminal:
```
pip install pykda
```

## Running the examples locally

To run the example notebooks locally, first clone the repository:
```
git clone https://github.com/JoostBerkhout/PyKDA
```
Then, make sure your Python version has poetry:
```
pip install --upgrade poetry
```
Now, go into the PyKDA repository and set up a virtual environment. 
We want a virtual environment that also contains all dependencies needed to run 
the example notebooks, so we also need to install the optional examples 
dependency group. This goes like so:
```
cd PyKDA
poetry install --with examples
```
This might take a few minutes to resolve, but only needs to be done once. 
After setting everything up, simply open the jupyter notebooks:
```
poetry run jupyter notebook
```
This will open up a page in your browser, where you can navigate to the example 
notebooks in the examples/ folder!

!!! warning
    The plotting of the Markov chain using `pyvis` does not always work when 
    running the notebooks locally. When this happens, try to put `notebook=False`
    in the `plot` functions as argument.

!!! note
    The above instructions also allow to change the code locally.
