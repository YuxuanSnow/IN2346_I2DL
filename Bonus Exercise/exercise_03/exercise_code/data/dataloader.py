"""Definition of Dataloader"""

import numpy as np


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        ########################################################################
        # TODO:                                                                #
        # Define an iterable function that samples batches from the dataset.   #
        # Each batch should be a dict containing numpy arrays of length        #
        # batch_size (except for the last batch if drop_last=True)             #
        # Hints:                                                               #
        #   - np.random.permutation(n) can be used to get a list of all        #
        #     numbers from 0 to n-1 in a random order                          #
        #   - To load data efficiently, you should try to load only those      #
        #     samples from the dataset that are needed for the current batch.  #
        #     An easy way to do this is to build a generator with the yield    #
        #     keyword, see https://wiki.python.org/moin/Generators             #
        #   - Have a look at the "DataLoader" notebook first                   #
        ########################################################################

        usable_set = (len(self.dataset) // self.batch_size) * self.batch_size

        if self.shuffle:
            data_set = np.random.permutation(len(self.dataset))
            if self.drop_last:
                data_set = data_set[0:usable_set]
            index_iterator = iter(data_set)  # define indices as iterator
        else:
            data_set = range(len(self.dataset))
            if self.drop_last:
                data_set = data_set[0:usable_set]
            index_iterator = iter(data_set)  # define indices as iterator

        def combine_batch_dicts(batch):
            batch_dict = {}
            for data_dict in batch:
                for key, value in data_dict.items():
                    if key not in batch_dict:
                        batch_dict[key] = []
                    batch_dict[key].append(value)
            return batch_dict

        batch = []

        for index in index_iterator:  # iterate over indices using the iterator
            batch.append(self.dataset[index])
            if len(batch) == self.batch_size:
                yield combine_batch_dicts(batch)  # use yield keyword to define a iterable generator
                batch = []
        if (self.drop_last is False) and (len(self.dataset) % self.batch_size != 0):
            yield combine_batch_dicts(batch)
        '''
        for index in index_iterator:  # iterate over indices using the iterator
            batch.append(self.dataset[index])
            if len(batch) == self.batch_size:
                yield batch  # use yield keyword to define a iterable generator
                batches.append(batch)
                batch = []

        if (self.drop_last is False) and (len(self.dataset) % self.batch_size != 0):
            batches.append(batch)
        '''
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def __len__(self):
        length = None
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataloader                                  #
        # Hint: this is the number of batches you can sample from the dataset  #
        ########################################################################

        if self.drop_last:
            length = (len(self.dataset) // self.batch_size)
        elif len(self.dataset) % self.batch_size == 0:
            length = (len(self.dataset) // self.batch_size)
        else:
            length = (len(self.dataset) // self.batch_size) + 1
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return length
