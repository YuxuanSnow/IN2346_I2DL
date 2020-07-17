"""Tests for ImageFolderDataset in data/image_folder_dataset.py"""

import random

from .base_tests import UnitTest, MethodTest, ClassTest, test_results_to_score
from .len_tests import LenTest


class MakeDatasetTestImageType(UnitTest):
    """Test whether make_dataset() loads paths only and not actual images"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.image_type = str
        self.wrong_image_type = None

    def test(self):
        images, _ = self.dataset.make_dataset(
            directory=self.dataset.root_path,
            class_to_idx=self.dataset.class_to_idx,
        )
        assert len(images) > 100
        random_indices = random.sample(range(len(images)), 100)
        for i in random_indices:
            if not isinstance(images[i], self.image_type):
                self.wrong_image_type = type(images[i])
                return False
        return True

    def define_failure_message(self):
        return "Expected images to contain file paths only, but got type %s" \
               % self.wrong_image_type


class MakeDatasetTest(MethodTest):
    """Test make_dataset() method of ImageFolderDataset"""
    def define_tests(self, dataset):
        return [
            MakeDatasetTestImageType(dataset),
        ]

    def define_method_name(self):
        return "make_dataset"


class GetItemTestType(UnitTest):
    """Test whether __getitem()__ returns correct data type"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.type = dict
        self.wrong_type = None

    def test(self):
        random_indices = random.sample(range(len(self.dataset)), 100)
        for i in random_indices:
            if not isinstance(self.dataset[i], self.type):
                self.wrong_type = type(self.dataset[i])
                return False
        return True

    def define_failure_message(self):
        return "Expected __getitem()__ to return type %s but got %s." \
               % (self.type, self.wrong_type)


class GetItemTestImageShape(UnitTest):
    """Test whether images loaded by __getitem__() are of correct shape"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.expected_shape = (32, 32, 3)
        self.wrong_shape = None

    def test(self):
        random_indices = random.sample(range(len(self.dataset)), 100)
        for i in random_indices:
            if self.dataset[i]["image"].shape != self.expected_shape:
                self.wrong_shape = self.dataset[i]["image"].shape
                return False
        return True

    def define_failure_message(self):
        return "Expected images to have shape %s but got %s." \
               % (str(self.expected_shape), str(self.dataset.images.shape))


class GetItemTestOrder(UnitTest):
    """Test whether order of items is correct"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.test_indices = [1, 3, 42, 100, 333, 999, 4242, 9999, 33333, -1]
        self.expected_labels = [0, 0, 0, 0, 0, 0, 0, 2, 6, 9]

    def test(self):
        labels = [self.dataset[index]["label"] for index in self.test_indices]
        return labels == self.expected_labels

    def define_failure_message(self):
        return "Order of items loaded by __getitem()__ not correct."


class GetItemTest(MethodTest):
    """Test __getitem__() method of ImageFolderDataset"""
    def define_tests(self, dataset):
        return [
            GetItemTestType(dataset),
            GetItemTestOrder(dataset),
            GetItemTestImageShape(dataset),
        ]

    def define_method_name(self):
        return "__getitem__"


class ImageFolderDatasetTest(ClassTest):
    """Test class ImageFolderDataset"""
    def define_tests(self, dataset):
        return [
            MakeDatasetTest(dataset),
            LenTest(dataset, 50000),
            GetItemTest(dataset),
        ]

    def define_class_name(self):
        return "ImageFolderDataset"


def test_image_folder_dataset(dataset):
    """Test class ImageFolderDataset"""
    test = ImageFolderDatasetTest(dataset)
    return test_results_to_score(test())
