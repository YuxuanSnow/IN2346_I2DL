import numpy as np


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

    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_loss_history and solver.val_loss_history will be lists containing
    the losses of the model on the training and validation set at each epoch.
    """

    def __init__(self, model, train_dataloader, val_dataloader, loss_func,
                 learning_rate, optimizer, verbose=True, print_every=1):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above

        - train_dataloader: A generator object returning training data
        - val_dataloader: A generator object returning validation data

        - loss_func: Loss function object.
        - learning_rate: Float, learning rate used for gradient descent.

        - optimizer: The optimizer specifying the update rule

        Optional arguments:
        - verbose: Boolean; if set to false then no output will be printed during
          training.
        - print_every: Integer; training losses will be printed every print_every
          iterations.
        """
        self.model = model
        self.learning_rate = learning_rate
        self.loss_func = loss_func

        self.opt = optimizer(model, loss_func, learning_rate)

        self.verbose = verbose
        self.print_every = print_every

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.best_val_loss = None
        self.best_params = None

        self.train_loss_history = []
        self.val_loss_history = []

        self.train_batch_loss = []
        self.val_batch_loss = []

        self.num_operation = 0

    def _step(self, X, y, validation=False):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.

        :param X: batch of training features
        :param y: batch of corresponding training labels
        :param validation: Boolean indicating whether this is a training or
            validation step

        :return loss: Loss between the model prediction for X and the target
            labels y
        """
        loss = None

        # Forward pass
        y_pred = self.model.forward(X)
        # Compute loss
        loss = self.loss_func.forward(y_pred, y)

        # Count number of operations
        self.num_operation += self.model.num_operation

        # Perform gradient update (only in train mode)
        if not validation:
            # Compute gradients
            self.opt.backward(y_pred, y)
            # Update weights
            self.opt.step()

            # If it was a training step, we need to count operations for
            # backpropagation as well
            self.num_operation += self.model.num_operation

        return loss

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

    def train(self, epochs=100):
        """
        Run optimization to train the model.
        """

        # Start an epoch
        for t in range(epochs):

            # Iterate over all training samples
            train_epoch_loss = 0.0

            for batch in self.train_dataloader:
                # Unpack data
                X = batch['features']
                y = batch['target']

                # Update the model parameters.
                validate = t == 0
                train_loss = self._step(X, y, validation=validate)

                self.train_batch_loss.append(train_loss)
                train_epoch_loss += train_loss

            train_epoch_loss /= len(self.train_dataloader)

            # Iterate over all validation samples
            val_epoch_loss = 0.0

            for batch in self.val_dataloader:
                # Unpack data
                X = batch['features']
                y = batch['target']

                # Update the model parameters.
                val_loss = self._step(X, y, validation=True)
                self.val_batch_loss.append(val_loss)
                val_epoch_loss += val_loss

            val_epoch_loss /= len(self.val_dataloader)

            # Record the losses for later inspection.
            self.train_loss_history.append(train_epoch_loss)
            self.val_loss_history.append(val_epoch_loss)

            if self.verbose and t % self.print_every == 0:
                print('(Epoch %d / %d) train loss: %f; val loss: %f' % (
                    t + 1, epochs, train_epoch_loss, val_epoch_loss))

            # Keep track of the best model
            self.update_best_loss(val_epoch_loss)

        # At the end of training swap the best params into the model
        self.model.params = self.best_params

    def get_dataset_prediction(self, loader):
        prediction = []
        for batch in loader:
            X = batch['features']
            y_pred = self.model.forward(X)
            prediction.append(y_pred)
        return np.concatenate(prediction, axis=0)

    def update_best_loss(self, val_loss):
        # Update the model and best loss if we see improvements.
        if not self.best_val_loss or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_params = self.model.params
