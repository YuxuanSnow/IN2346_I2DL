import random
from math import log10
from itertools import product
from exercise_code.solver import Solver
from exercise_code.networks.layer import Sigmoid, Tanh, LeakyRelu, Relu
from exercise_code.networks.optimizer import SGD, Adam
from exercise_code.networks import (ClassificationNet, BCE,
                                    CrossEntropyFromLogits)


ALLOWED_RANDOM_SEARCH_PARAMS = ['log', 'int', 'float', 'item']


def grid_search(train_loader, val_loader,
                grid_search_spaces = {
                    "learning_rate": [0.0001, 0.001, 0.01, 0.1],
                    "reg": [1e-4, 1e-5, 1e-6]
                },
                model_class=ClassificationNet, epochs=20, patience=5):
    """
    A simple grid search based on nested loops to tune learning rate and
    regularization strengths.
    Keep in mind that you should not use grid search for higher-dimensional
    parameter tuning, as the search space explodes quickly.

    Required arguments:
        - train_dataloader: A generator object returning training data
        - val_dataloader: A generator object returning validation data

    Optional arguments:
        - grid_search_spaces: a dictionary where every key corresponds to a
        to-tune-hyperparameter and every value contains a list of possible
        values. Our function will test all value combinations which can take
        quite a long time. If we don't specify a value here, we will use the
        default values of both our chosen model as well as our solver
        - model: our selected model for this exercise
        - epochs: number of epochs we are training each model
        - patience: if we should stop early in our solver

    Returns:
        - The best performing model
        - A list of all configurations and results
    """
    configs = []

    """
    # Simple implementation with nested loops
    for lr in grid_search_spaces["learning_rate"]:
        for reg in grid_search_spaces["reg"]:
            configs.append({"learning_rate": lr, "reg": reg})
    """

    # More general implementation using itertools
    for instance in product(*grid_search_spaces.values()):
        configs.append(dict(zip(grid_search_spaces.keys(), instance)))

    return findBestConfig(train_loader, val_loader, configs, epochs, patience,
                          model_class)



def random_search(train_loader, val_loader,
                  random_search_spaces = {
                      "learning_rate": ([0.0001, 0.1], 'log'),
                      "hidden_size": ([100, 400], "int"),
                      "activation": ([Sigmoid(), Relu()], "item"),
                  },
                  model_class=ClassificationNet, num_search=20, epochs=20,
                  patience=5):
    """
    Samples N_SEARCH hyper parameter sets within the provided search spaces
    and returns the best model.

    See the grid search documentation above.

    Additional/different optional arguments:
        - random_search_spaces: similar to grid search but values are of the
        form
        (<list of values>, <mode as specified in ALLOWED_RANDOM_SEARCH_PARAMS>)
        - num_search: number of times we sample in each int/float/log list
    """
    configs = []
    for _ in range(num_search):
        configs.append(random_search_spaces_to_config(random_search_spaces))

    return findBestConfig(train_loader, val_loader, configs, epochs, patience,
                          model_class)


def findBestConfig(train_loader, val_loader, configs, EPOCHS, PATIENCE,
                   model_class):
    """
    Get a list of hyperparameter configs for random search or grid search,
    trains a model on all configs and returns the one performing best
    on validation set
    """
    
    best_val = None
    best_config = None
    best_model = None
    results = []
    
    for i in range(len(configs)):
        print("\nEvaluating Config #{} [of {}]:\n".format(
            (i+1), len(configs)),configs[i])

        model = model_class(**configs[i])
        solver = Solver(model, train_loader, val_loader, **configs[i])
        solver.train(epochs=EPOCHS, patience=PATIENCE)
        results.append(solver.best_model_stats)

        if not best_val or solver.best_model_stats["val_loss"] < best_val:
            best_val, best_model,\
            best_config = solver.best_model_stats["val_loss"], model, configs[i]
            
    print("\nSearch done. Best Val Loss = {}".format(best_val))
    print("Best Config:", best_config)
    return best_model, list(zip(configs, results))
        

def random_search_spaces_to_config(random_search_spaces):
    """"
    Takes search spaces for random search as input; samples accordingly
    from these spaces and returns the sampled hyper-params as a config-object,
    which will be used to construct solver & network
    """
    
    config = {}

    for key, (rng, mode)  in random_search_spaces.items():
        if mode not in ALLOWED_RANDOM_SEARCH_PARAMS:
            print("'{}' is not a valid random sampling mode. "
                  "Ignoring hyper-param '{}'".format(mode, key))
        elif mode == "log":
            if rng[0] <= 0 or rng[-1] <=0:
                print("Invalid value encountered for logarithmic sampling "
                      "of '{}'. Ignoring this hyper param.".format(key))
                continue
            sample = random.uniform(log10(rng[0]), log10(rng[-1]))
            config[key] = 10**(sample)
        elif mode == "int":
            config[key] = random.randint(rng[0], rng[-1])
        elif mode == "float":
            config[key] = random.uniform(rng[0], rng[-1])
        elif mode == "item":
            config[key] = random.choice(rng)

    return config