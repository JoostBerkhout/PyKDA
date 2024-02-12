import numpy as np
import numpy.linalg as la


class MarkovChain:
    """
    Capture a Markov chain with transition matrix P.

    Parameters
    ----------
    P : np.ndarray
        Probability transition matrix.

    """

    def __init__(self, P: np.ndarray):
        self.P = P
        self.n = P.shape[0]
        self.I = np.eye(self.n)
        self.bUniChain = True

    def pi(self):
        """Stationary distribution (if it exists). """

        if self.bUniChain:
            # stationary distribution exists
            Z = self.P - self.I
            Z[:, [0]] = np.ones((self.n, 1))
            # TODO: instead of inverse, solve a system of linear equations
            return la.inv(Z)[[0], :]
        else:
            raise Warning('Stationary distribution does not exist.')


if __name__ == '__main__':

    P = np.array(
        [[0.    ,     0.52493781, 0.47506219],
         [0.       ,  0.66666667 ,0.33333333],
         [0.06666667 ,0.33333333 ,0.6       ]]
        )
    MC = MarkovChain(P)

    print(MC.pi())