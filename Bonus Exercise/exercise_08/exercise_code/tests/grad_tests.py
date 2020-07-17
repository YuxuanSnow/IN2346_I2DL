import numpy as np
from .base_tests import UnitTest, CompositeTest, test_results_to_score
import math
from .gradient_check_utils import eval_numerical_gradient_array

epsilon = 1e-7


class TestForwardPass(UnitTest):
    """Test whether Sigmoid of 0 is correct"""

    def __init__(self, x, gamma, beta, bn_params):
        self.x = x
        self.gamma = gamma
        self.beta = beta
        self.bn_params = bn_params
        self.bn = layer

    def test(self):
                
        return self.value == 0.5

    def define_failure_message(self):
        return "Sigmoid of 0 is incorrect. Expected: 0.5 Evaluated: " + str(self.value)



class RunAllBatchNormTests(CompositeTest):

    def define_tests(self, layer):
        return [
            BatchNormForward(layer),
        ]

    def define_success_message(self):
        return "Congratulations you have passed all the unit tests!!!"

    def define_failure_message(self):
        return "Test cases are still failing!"


class BatchNormTestWrapper:
    def __init__(self, layer):
        self.bn_tests = RunAllBatchNormTests(model)

    def __call__(self, *args, **kwargs):
        return "You secured a score of :" + str(test_results_to_score(self.sigmoid_tests()))

