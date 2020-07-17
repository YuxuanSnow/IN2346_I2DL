import pickle
import os


def save_pickle(data_dict, file_name):
    """Save given data dict to pickle file file_name in models/"""
    directory = 'models'
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(data_dict, open(os.path.join(directory, file_name), 'wb', 4))
