import numpy as np


def forward(A, B, pi, O):
    """
    Calculates the probability of an observation sequence O given the model(A, B, pi).
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities (N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The probability of the observation sequence and the calculated alphas in the Trellis diagram with shape
             (N, T) which should be a numpy array.
    """

    alpha = np.zeros((O.shape[0], A.shape[0]))

    for i, value in enumerate(pi):
        alpha[0][i] = value * B[i][O[0]]

    for t in range(1, O.shape[0]):
        for j in range(A.shape[0]):
            alpha[t, j] = alpha[t - 1].dot(A[:, j]) * B[j, O[t]]
    updated = np.transpose(alpha)
    p = sum(map(lambda x: x[-1], updated))
    return p, updated


def viterbi(A, B, pi, O):
    """
    Calculates the most likely state sequence given model(A, B, pi) and observation sequence.
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities(N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The most likely state sequence with shape (T,) and the calculated deltas in the Trellis diagram with shape
             (N, T). They should be numpy arrays.
    """

    probs = np.zeros((A.shape[0], O.shape[0]))
    values = np.zeros((A.shape[0], O.shape[0] - 1))
    for i, value in enumerate(pi):
        probs[i][0] = value * B[i][O[0]]

    for j in range(1, O.shape[0]):
        for i in range(A.shape[0]):
            mult = A[:, i] * probs[:, j - 1]
            probs[i, j] = np.max(mult) * B[i, O[j]]
            values[i, j - 1] = np.argmax(mult)

    seq = np.zeros(O.shape[0])
    seq[-1] = np.argmax(probs[:, -1])
    for i in range(O.shape[0] - 2, -1, -1):
        seq[i] = values[int(seq[i + 1]), i]

    return seq, probs