import pickle
import os

import numpy as np


def evaluate(x):
    sum_exp = np.sum(x)
    if sum_exp > 4.53:
        print('Hurray, you passed!! Now save your model and submit it!')
        return 75
    else:
        print('I think you can do better...')
        return 0


def save_pickle(data_dict, file_name):
    """Save given data dict to pickle file file_name in models/"""
    directory = 'models'
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(data_dict, open(os.path.join(directory, file_name), 'wb', 4))
