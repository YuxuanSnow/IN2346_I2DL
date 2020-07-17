import os
import torch
import pickle

from exercise_code.MyPytorchModel import MyPytorchModel


PARAM_LIMIT = 5e6
SIZE_LIMIT_MB = 20
ACC_THRESHOLD = 0.5

def checkParams(model):
    
    n_params = sum(p.numel() for p in model.parameters())
    
    if n_params > PARAM_LIMIT: 
        print("Your model has {:.3f} mio. params but must have less than 5 mio. params. Simplify your model before submitting it. You won't need that many params :)".format(n_params / 1e6))
        return False

    print("FYI: Your model has {:.3f} params.".format(n_params / 1e6))
    return True
        
def checkLayers(model):
    
    '''
        Important Note: convolutional layers are not allowed in this exercise, as they have not been covered yet in the lecture.
        Using these would be highly unfair towards student that haven't heard about them yet. 
    '''
    
    forbidden_layers = [torch.nn.modules.conv.Conv2d]
    
    for key, module in model._modules.items():
        for i in range(len(module)):
            if type(module[i]) == forbidden_layers:
                print("Please don't use convolutions! For now, only use layers that have been already covered in the lecture!")
                return False
            
    return True

def checkSize(path = "./models/cifar_pytorch.torch"):
    size = os.path.getsize(path)
    sizeMB = size / 1e6
    if sizeMB > SIZE_LIMIT_MB:
        print("Your model is too large! The size is {:.1f} MB, but it must be less than 20 MB. Please simplify your model before submitting.".format(sizeMB))
        return False
    print("Great! Your model size is less than 20 MB and will be accepted :)")
    return True


def printModelInfo(model):
    accepted = checkParams(model) & checkLayers(model)
    print("Model accepted!") if accepted else print("Model not accepted. Please follow the instructions.")
    return accepted

def load_model(model_path):
    model_dict = pickle.load(open(model_path, 'rb'))["cifar_classification_ptl"]
    model = MyPytorchModel(model_dict["hparams"])
    model.load_state_dict(model_dict["state_dict"])
    return model

def save_model(model, file_name, directory = "models"):
    model_dict = {"cifar_classification_ptl":{"state_dict":model.state_dict(), "hparams": model.hparams}}
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(model_dict, open(os.path.join(directory, file_name), 'wb', 4))
    
def test_and_save(model):
        
    _, val_acc = model.getTestAcc(model.val_dataloader())
    print("Validation-Accuracy: {}%".format(val_acc))
    if val_acc < ACC_THRESHOLD:
        print("That's too low! Please tune your model in order to reach at least {}% before running on the test set and submitting!".format(ACC_THRESHOLD * 100))
        return
    
    if not (checkParams(model) & checkLayers(model)):
        return
    
    save_model(model, "cifar_pytorch.p")
    if not checkSize("./models/cifar_pytorch.p"):
        return
    
    print("Your model has been saved and is ready to be submitted. NOW, let's check the test-accuracy.")
    _, test_acc = model.getTestAcc()
    print("Test-Accuracy: {}%".format(test_acc))
        
        
    