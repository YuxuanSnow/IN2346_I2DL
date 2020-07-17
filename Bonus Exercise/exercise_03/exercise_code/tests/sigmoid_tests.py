import numpy as np
from .base_tests import UnitTest, CompositeTest, test_results_to_score
import math

epsilon = 1e-7


class Sigmoid_Of_Zero(UnitTest):
    """Test whether Sigmoid of 0 is correct"""

    def __init__(self, model):
        self.value = model.forward(np.float(0))

    def test(self):
        return self.value == 0.5

    def define_failure_message(self):
        return "Sigmoid of 0 is incorrect. Expected: 0.5 Evaluated: " + str(self.value)


class Sigmoid_Of_Zero_Array(UnitTest):
    """Test whether Sigmoid of a numpy array [0, 0, 0, 0, 0] is correct"""

    def __init__(self, model):
        self.value = model.forward(np.asarray([0, 0, 0, 0, 0])).sum()

    def test(self):
        return self.value == 5 * 0.5

    def define_failure_message(self):
        return "The array values do not sum up to expected result. Expected: 2.5 Evaluated:" + str(self.value)


class Sigmoid_Of_100(UnitTest):
    """Test whether Sigmoid of 100 is correct"""

    def __init__(self, model):
        self.value = model.forward(np.float(100))

    def test(self):
        return math.fabs(1 - self.value) < epsilon

    def define_failure_message(self):
        return "At an input of 100 value should be ~1. Expected ~ 1 Evaluated: " + str(self.value)


class Sigmoid_Of_Array_of_100(UnitTest):
    """Test whether Sigmoid of [100, 100, 100, 100, 100] is correct"""

    def __init__(self, model):
        self.value = model.forward(np.asarray([100, 100, 100, 100, 100])).sum()

    def test(self):
        return math.fabs(5 - self.value) < epsilon

    def define_failure_message(self):
        return "At an input of 100 value should be ~1. Expected ~5 Evaluated: " + str(self.value)


class RunAllSigmoidTests(CompositeTest):

    def define_tests(self, model):
        return [
            Sigmoid_Of_Zero(model),
            Sigmoid_Of_Zero_Array(model),
            Sigmoid_Of_100(model),
            Sigmoid_Of_Array_of_100(model),
        ]

    def define_success_message(self):
        return "Congratulations you have passed all the unit tests!!!"

    def define_failure_message(self):
        return "Test cases are still failing!"


class SigmoidTestWrapper:
    def __init__(self, model):
        self.sigmoid_tests = RunAllSigmoidTests(model)

    def __call__(self, *args, **kwargs):
        return "You secured a score of :" + str(test_results_to_score(self.sigmoid_tests()))

