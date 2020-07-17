import numpy as np


from exercise_code.networks.optimizer import Optimizer

class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training classification
    or regression models.
    The Solver performs gradient descent using the given learning rate.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    To train a model, you will first construct a Solver instance, passing the
    model, dataset, learning_rate to the constructor.
    You will then call the train() method to run the optimization 
    procedure and train the model.

    After the train() method returns, model.W will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_loss_history and solver.val_loss_history will be lists containing
    the losses of the model on the training and validation set at each epoch.
    """

    def __init__(self, model, data, loss_func, learning_rate,
                 is_regression=True, verbose=True, print_every=100):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above

        - data: A dictionary of training and validation data with the following:
          'X_train': Training input samples.
          'X_val':   Validation input samples.
          'y_train': Training labels.
          'y_val':   Validation labels.

        - loss_func: Loss function object.
        - learning_rate: Float, learning rate used for gradient descent.

        Optional arguments:
        - verbose: Boolean; if set to false then no output will be printed during
          training.
        - print_every: Integer; training losses will be printed every print_every
          iterations.
        """
        self.model = model
        self.learning_rate = learning_rate
        self.loss_func = loss_func

        # Use an `Optimizer` object to do gradient descent on our model.
        self.opt = Optimizer(model, learning_rate)

        self.is_regression = is_regression
        self.verbose = verbose
        self.print_every = print_every

        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.best_val_loss = None
        self.best_W = None

        self.train_loss_history = []
        self.val_loss_history = []

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        model = self.model
        loss_func = self.loss_func
        X_train = self.X_train
        y_train = self.y_train
        opt = self.opt
        ###########################################################################
        #   TODO: 													              #
        #	Get the gradients dhat{y} / dW and dLoss / dhat{y}. 	              #
        #   Combine them via the chain rule to obtain dLoss / dW.                 #
        #   Proceed by performing an optimizing step using the given              #
        #   optimizer (by calling opt.step() with the gradient wrt W)             #
        #                                                                         #
        #   Hint: don't forget to divide number of samples when computing the     #
        #   gradient!                                                             #
        ###########################################################################
        model.train()
        model_forward, model_backward = model(X_train)
        loss, loss_grad = loss_func(model_forward, y_train)
        grad = loss_grad * model_backward
        grad = np.mean(grad, 0, keepdims=True)
        opt.step(grad.T)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################  

    def check_loss(self, validation=True):
        """
        Check loss of the model on the train/validation data.

        Returns:
        - loss: Averaged loss over the relevant samples.
        """

        X = self.X_val if validation else self.X_train
        y = self.y_val if validation else self.y_train

        model_forward, _ = self.model(X)
        loss, _ = self.loss_func(model_forward, y)

        return loss.mean()

    def train(self, epochs = 1000):
        """
        Run optimization to train the model.
        """


        for t in range(epochs):
            # Update the model parameters.
            self._step()

            # Check the performance of the model.
            train_loss = self.check_loss(validation=False)
            val_loss = self.check_loss(validation=True)

            # Record the losses for later inspection.
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            if self.verbose and t % self.print_every == 0:
                print('(Epoch %d / %d) train loss: %f; val_loss: %f' % (
                        t, epochs, train_loss, val_loss))

            # Keep track of the best model
            self.update_best_loss(val_loss)

        # At the end of training swap the best params into the model
        self.model.W = self.best_W

    def update_best_loss(self, val_loss):
        # Update the model and best loss if we see improvements.
        if not self.best_val_loss or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_W = self.model.W
