
import numpy as np

from .base_tests import UnitTest, CompositeTest, MethodTest, ClassTest, test_results_to_score
from exercise_code.networks.loss import MSE
from exercise_code.networks.regression_net import RegressionNet
from .gradient_check import eval_numerical_gradient_array


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


class RegressionNetForwardTest(UnitTest):
    def __init__(self, model):
        self.error = 0.0
        self.model = model

    def test(self):
        N, D, H = 3, 5, 50

        # model = RegressionNet(input_size=D, hidden_size=H)
        model = self.model(input_size=D, hidden_size=H)

        model.params['W1'] = np.linspace(-0.7, 0.3, num=D * H).reshape(D, H)
        model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
        model.params['W2'] = np.linspace(-0.3, 0.4, num=H).reshape(H, 1)
        model.params['b2'] = np.linspace(-0.9, 0.1, num=1)
        X = np.linspace(-5.5, 4.5, num=N * D).reshape(D, N).T
        y_pred = model.forward(X)
        correct_y = np.asarray([[1.5916902],
                                [1.59461811],
                                [1.61232696]])
        self.error = np.abs(y_pred - correct_y).sum()

        return self.error < 1e-6

    def define_failure_message(self):
        return "RegressionNet forward incorrect. Expected: < 1e-6 Evaluated: " + str(self.error)


class RegressionNetBackwardTest(UnitTest):
    def __init__(self, model):
        self.error = {
            'W1': 0.0,
            'b1': 0.0,
            'W2': 0.0,
            'b2': 0.0
        }
        self.model = model

    def test(self):
        N, D, H = 3, 5, 50

        # model = RegressionNet(input_size=D, hidden_size=H)
        model = self.model(input_size=D, hidden_size=H)
        criterion = MSE()

        X = np.linspace(-5.5, 4.5, num=N * D).reshape(D, N).T
        target = np.array([[1.]])

        model.params['W1'] = np.linspace(-0.7, 0.3, num=D * H).reshape(D, H)
        model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
        model.params['W2'] = np.linspace(-0.3, 0.4, num=H).reshape(H, 1)
        model.params['b2'] = np.linspace(-0.9, 0.1, num=1)

        y_pred = model.forward(X)
        dy = criterion.backward(y_pred, target)
        grads = model.backward(dy)

        failed = False
        for name in sorted(grads):
            f = lambda _: model.forward(X)
            grad_num = eval_numerical_gradient_array(f, model.params[name], dy) / N
            error = rel_error(grad_num, grads[name])
            self.error[name] = error
            if error > 1e-5:
                failed = True

        return not failed

    def define_failure_message(self):
        msg = "RegressionNet forward incorrect."
        for name in sorted(self.error):
            msg += f"\n\tExpected < 1e-5: Evaluated: {self.error[name]}"
        return msg


class ForwardTest(MethodTest):
    """Test forward() method of RegressionNet"""

    def define_tests(self, regression_net):
        return [
            RegressionNetForwardTest(regression_net)
        ]

    def define_method_name(self):
        return "forward"


class BackwardTest(MethodTest):
    """Test backward() method of RegressionNet"""

    def define_tests(self, regression_net):
        return [
            RegressionNetBackwardTest(regression_net)
        ]

    def define_method_name(self):
        return "backward"


class RegressionNetTest(ClassTest):
    """Test RegressionNet class"""

    def define_tests(self, regression_net):
        return [
            ForwardTest(regression_net),
            RegressionNetBackwardTest(regression_net),
        ]

    def define_class_name(self):
        return "RegressionNet"


def test_regression_net(
        regression_net
):
    """Test RegressionNet class"""
    test = RegressionNetTest(
        regression_net
    )
    return test_results_to_score(test())
