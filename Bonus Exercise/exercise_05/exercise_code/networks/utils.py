import numpy as np


def binarize(X, y, a_percentile, b_percentile):
    """ Splits data to be smaller than the a_percentil and larger than b_percentile
    :param x: input
    :param y: labels
    :param a_percentile:
    :param b_percentile:
    :return:
    :rtype: X, Y
    """
    data_index = ((a_percentile >= y) | (y >= b_percentile))
    y = y[data_index]
    x = X[data_index[:, 0]]

    y[y <= a_percentile] = 0
    y[y >= b_percentile] = 1

    return x, np.expand_dims(y, 1)


def test_accuracy(y_pred, y_true):
    """ Compute test error / accuracy
    Params:
    ------
    y_pred: model prediction
    y_true: ground truth values
    return:
    ------
    Accuracy / error on test set
    """

    # Apply threshold
    threshold = 0.50

    y_binary = np.zeros_like((y_pred))
    y_binary[y_pred >= threshold] = 1
    y_binary[y_pred < threshold] = 0

    # Get final predictions.
    y_binary = y_binary.flatten().astype(int)
    y_true = y_true.flatten().astype(int)

    acc = (y_binary == y_true).mean()
    return acc
