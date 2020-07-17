import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20, activation="tanh"):
        super(RNN, self).__init__()
        
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        ############################################################################
        # TODO: Build a simple one layer RNN with an activation with the attributes#
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h as 0 if these values are not given.                          #
        ############################################################################

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation

        self.W = nn.Linear(input_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []

        #######################################################################
        #  TODO: Perform the forward pass                                     #
        #######################################################################   

        # read how many samples
        T = x.size()[0]

        # initialize h with 0. h has two dimension. one is which sample and other is which "row" of current sample
        h = torch.zeros((x.size()[1], self.hidden_size))

        # for every sample in the set
        for t in range(T):
            x_t = x[t, :, :]
            h = self.tanh(self.W(x_t) + self.V(h))
            h_seq.append(h)
        h_seq = torch.stack(tuple(h_seq))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return h_seq , h

class LSTM(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20):
        super(LSTM, self).__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        """
        ############################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes      #
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h and c as 0 if these values are not given.                    #

        ############################################################################
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_forget = nn.Linear(input_size, hidden_size)
        self.U_forget = nn.Linear(hidden_size, hidden_size)
        self.W_in = nn.Linear(input_size, hidden_size)
        self.U_in = nn.Linear(hidden_size, hidden_size)
        self.W_out = nn.Linear(input_size, hidden_size)
        self.U_out = nn.Linear(hidden_size, hidden_size)
        self.W_cell = nn.Linear(input_size, hidden_size)
        self.U_cell = nn.Linear(hidden_size, hidden_size)

    ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################       


    def forward(self, x, h=None , c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """

        #######################################################################
        #  TODO: Perform the forward pass                                     #
        #######################################################################   

        seq_len, batch_size, input_size = x.size()
        if h is None:
            h = torch.zeros((1, batch_size, self.hidden_size))
        if c is None:
            c = torch.zeros((1, batch_size, self.hidden_size))
        h_seq = []
        for t in range(seq_len):
            x_t = x[t, :, :]
            forget_t = torch.sigmoid(self.W_forget(x_t) + self.U_forget(h))
            in_t = torch.sigmoid(self.W_in(x_t) + self.U_in(h))
            out_t = torch.sigmoid(self.W_out(x_t) + self.U_out(h))
            c = torch.mul(forget_t, c)
            c += torch.mul(in_t, torch.tanh(self.W_cell(x_t) + self.U_cell(h)))
            h = torch.mul(out_t, torch.tanh(c))
            h_seq.append(h)
        h_seq = torch.stack(tuple(h_seq)).reshape((seq_len, batch_size, self.hidden_size))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    
        return h_seq, (h, c)

