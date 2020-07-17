"""Tests for Transform classes in data/image_folder_dataset.py"""

import numpy as np

from .base_tests import UnitTest, MethodTest, ClassTest, test_results_to_score


class RescaleTransformTestMin(UnitTest):
    """Test whether RescaleTransform rescales to correct minimum value"""
    def __init__(self, dataset, min_):
        self.dataset = dataset
        self.min = min_
        self.min_val = None

    def test(self):
        self.min_val = float("inf")
        for sample in self.dataset:
            image_min = sample["image"].min()
            if image_min < self.min_val:
                self.min_val = image_min
        return self.min_val == self.min

    def define_failure_message(self):
        return "Images were not rescaled correctly. " \
               "Expected new minimum to be %d but got %s." \
               % (self.min, str(self.min_val))


class RescaleTransformTestMax(UnitTest):
    """Test whether RescaleTransform rescales to correct maximum value"""
    def __init__(self, dataset, max_):
        self.dataset = dataset
        self.max = max_
        self.max_val = None

    def test(self):
        self.max_val = float("-inf")
        for sample in self.dataset:
            image_max = sample["image"].max()
            if image_max > self.max_val:
                self.max_val = image_max
        return self.max_val == self.max

    def define_failure_message(self):
        return "Images were not rescaled correctly. " \
               "Expected new minimum to be %d but got %s." \
               % (self.max, str(self.max_val))


class RescaleTransformTest(ClassTest):
    """Test class RescaleTransform"""
    def define_tests(self, dataset, range_):
        min_, max_ = range_
        return [
            RescaleTransformTestMin(dataset, min_),
            RescaleTransformTestMax(dataset, max_),
        ]

    def define_class_name(self):
        return "RescaleTransform"


def test_rescale_transform(dataset, range_=(0, 1)):
    """Test class RescaleTransform"""
    test = RescaleTransformTest(dataset, range_)
    return test_results_to_score(test())


class CIFARImageMeanTest(UnitTest):
    """Test whether computed CIFAR-10 image mean is correct"""
    def __init__(self, mean):
        self.mean = mean
        self.expected_mean = np.array([0.49191375, 0.48235852, 0.44673872])

    def test(self):
        return np.sum(self.mean - self.expected_mean) < 1e-3

    def define_failure_message(self):
        return "Computed image mean values incorrect."


class CIFARImageStdTest(UnitTest):
    """Test whether computed CIFAR-10 image std is correct """
    def __init__(self, std):
        self.std = std
        self.expected_std = np.array([0.24706447, 0.24346213, 0.26147554])

    def test(self):
        return np.sum(self.std - self.expected_std) < 1e-3

    def define_failure_message(self):
        return "Computed image standard deviation values incorrect."


class CIFARImageMeanStdTest(MethodTest):
    """Test compute_image_mean_and_std() in data/image_folder_dataset.py"""

    def define_tests(self, mean, std):
        return [
            CIFARImageMeanTest(mean),
            CIFARImageStdTest(std),
        ]

    def define_method_name(self):
        return "compute_image_mean_and_std"


def test_compute_image_mean_and_std(mean, std):
    """Test compute_image_mean_and_std() in data/image_folder_dataset.py"""
    test = CIFARImageMeanStdTest(mean, std)
    return test_results_to_score(test())
