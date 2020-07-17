"""Tests for DataLoader in data/dataloader.py"""

import numpy as np

from .len_tests import LenTest
from .base_tests import UnitTest, MethodTest, ClassTest, test_results_to_score


def get_values_flat(iterable):
    """get all values from a DataLoader/Dataset as a flat list"""
    data = []
    for batch in iterable:
        for value in batch.values():
            if isinstance(value, (list, np.ndarray)):
                for val in value:
                    data.append(val)
            else:
                data.append(value)
    return data


class IterTestIterable(UnitTest):
    """Test whether __iter()__ is iterable"""
    def __init__(self, iterable):
        self.iterable = iterable

    def test(self):
        for _ in self.iterable:
            pass
        return True

    def define_exception_message(self, exception):
        return "Object is not iterable."


class IterTestItemType(UnitTest):
    """Test whether __iter()__ returns correct item type"""
    def __init__(self, iterable, item_type):
        self.iterable = iterable
        self.item_type = item_type
        self.wrong_type = None

    def test(self):
        for item in self.iterable:
            if not isinstance(item, self.item_type):
                self.wrong_type = type(item)
                return False
        return True

    def define_failure_message(self):
        return "Expected items to be of type %s, got %s instead" \
               % (self.item_type, str(type(self.wrong_type)))


class IterTestBatchSize(UnitTest):
    """Test whether __iter__() of DataLoader uses correct batch_size"""
    def __init__(self, dataloader, batch_size):
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.wrong_batch_size = -1

    def test(self):
        if self.batch_size is None:
            return True
        for batch in self.dataloader:
            for _, value in batch.items():
                if len(value) != self.batch_size:
                    self.wrong_batch_size = len(value)
                    return False
        return True

    def define_failure_message(self):
        return "Wrong batch size (expected %d, got %d)." \
               % (self.batch_size, self.wrong_batch_size)


class IterTestNumBatches(UnitTest):
    """Test whether __iter__() of DataLoader loads correct number of batches"""
    def __init__(self, dataloader, num_batches):
        self.dataloader = dataloader
        self.num_batches = num_batches
        self.num_batches_iter = -1

    def test(self):
        self.num_batches_iter = 0
        for _ in self.dataloader:
            self.num_batches_iter += 1
        return self.num_batches_iter == self.num_batches

    def define_failure_message(self):
        return "Wrong number of batches (expected %d, got %d)." \
               % (self.num_batches, self.num_batches_iter)


class IterTestValuesUnique(UnitTest):
    """Test whether __iter__() of DataLoader loads all values only once"""
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def test(self):
        data = get_values_flat(self.dataloader)
        return len(data) == len(set(data))

    def define_failure_message(self):
        return "Values loaded are not unique."


class IterTestValueRange(UnitTest):
    """Test whether __iter__() of DataLoader loads correct value range"""
    def __init__(self, dataloader, min_, max_):
        self.dataloader = dataloader
        self.min = min_
        self.max = max_
        self.min_iter = -1
        self.max_iter = -1

    def test(self):
        if self.min is None or self.max is None:
            return True
        data = get_values_flat(self.dataloader)
        self.min_iter = min(data)
        self.max_iter = max(data)
        return self.min_iter == self.min and self.max_iter == self.max

    def define_failure_message(self):
        return "Expected lowest and highest value to be %d and %d, " \
               "but got minimum value %d and maximum value %d." \
               % (self.min, self.max, self.min_iter, self.max_iter)


class IterTestShuffled(UnitTest):
    """Test whether __iter__() of DataLoader shuffles the data"""
    def __init__(self, dataloader, shuffle):
        self.dataloader = dataloader
        self.shuffle = shuffle

    def test(self):
        if not self.shuffle:
            return True
        data = get_values_flat(self.dataloader)
        return data != sorted(data)

    def define_failure_message(self):
        return "Data loaded seems to be not shuffled."


class IterTestNonDeterministic(UnitTest):
    """Test whether __iter__() of DataLoader shuffles the data"""
    def __init__(self, dataloader, shuffle):
        self.dataloader = dataloader
        self.shuffle = shuffle

    def test(self):
        if not self.shuffle:
            return True
        data1 = get_values_flat(self.dataloader)
        data2 = get_values_flat(self.dataloader)
        return data1 != data2

    def define_failure_message(self):
        return "Loading seems to be deterministic, even though shuffle=True."


class IterTest(MethodTest):
    """Test __iter__() method of DataLoader"""
    def define_tests(
            self, dataloader, batch_size, len_, min_val, max_val, shuffle
    ):
        return [
            IterTestIterable(dataloader),
            IterTestItemType(dataloader, dict),
            IterTestBatchSize(dataloader, batch_size),
            IterTestNumBatches(dataloader, len_),
            IterTestValuesUnique(dataloader),
            IterTestValueRange(dataloader, min_val, max_val),
            IterTestShuffled(dataloader, shuffle),
            IterTestNonDeterministic(dataloader, shuffle)
        ]

    def define_method_name(self):
        return "__iter__"


class DataLoaderTest(ClassTest):
    """Test DataLoader class"""
    def define_tests(
            self, dataloader, batch_size, len_, min_val, max_val, shuffle
    ):
        return [
            LenTest(dataloader, len_),
            IterTest(dataloader, batch_size, len_, min_val, max_val, shuffle),
        ]

    def define_class_name(self):
        return "DataLoader"


def test_dataloader(
        dataset,
        dataloader,
        batch_size=1,
        shuffle=False,
        drop_last=False
):
    """Test DataLoader class"""
    if drop_last:
        test = DataLoaderTest(
            dataloader,
            batch_size=batch_size,
            len_=len(dataset) // batch_size,
            min_val=None,
            max_val=None,
            shuffle=shuffle,
        )
    else:
        test = DataLoaderTest(
            dataloader,
            batch_size=None,
            len_=int(np.ceil(len(dataset) / batch_size)),
            min_val=min(get_values_flat(dataset)),
            max_val=max(get_values_flat(dataset)),
            shuffle=shuffle,
        )
    return test_results_to_score(test())
