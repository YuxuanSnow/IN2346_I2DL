import os
import torch
import pickle

#from exercise_code.MyPytorchModel import MyPytorchModel


PARAM_LIMIT = 5e6
SIZE_LIMIT_MB = 50
ACC_THRESHOLD = 0.4

def checkParams(model):
    
    n_params = sum(p.numel() for p in model.parameters())
    
    if n_params > PARAM_LIMIT: 
        print("Your model has {:.3f} mio. params but must have less than 5 mio. params. Simplify your model before submitting it. You won't need that many params :)".format(n_params / 1e6))
        return False

    print("FYI: Your model has {:.3f} params.".format(n_params / 1e6))
    return True


def checkSize(path = "./models/cifar_pytorch.torch"):
    size = os.path.getsize(path)
    sizeMB = size / 1e6
    if sizeMB > SIZE_LIMIT_MB:
        print("Your model is too large! The size is {:.1f} MB, but it must be less than 20 MB. Please simplify your model before submitting.".format(sizeMB))
        return False
    print("Great! Your model size is less than 50 MB and will be accepted :)")
    return True

    
def test(acc):

    print("Validation-Accuracy: {}%".format(acc*100))
    if acc < ACC_THRESHOLD:
        print("That's too low! Please tune your model in order to reach at least {}% before running on the test set and submitting!".format(ACC_THRESHOLD * 100))

    else:
        print("Congrats! The accuracy passes the threshold, you can try to submit your model to server now.")
        
        
    