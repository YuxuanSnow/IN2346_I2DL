"""Utils for model saving"""

import os
import pickle


def save_model(model, file_name, directory="models"):
    """Save model as pickle"""
    model = model.cpu()
    model_dict = {
        "state_dict": model.state_dict(),
        "hparams": model.hparams
    }
    if not os.path.exists(directory):
        os.makedirs(directory)
    model_path = os.path.join(directory, file_name)
    pickle.dump(model_dict, open(model_path, 'wb', 4))
    print ("...Your model is saved to {} successfully!!".format(model_path))
    return model_path
