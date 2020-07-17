
import numpy as np

from .base_tests import UnitTest, CompositeTest, MethodTest, test_results_to_score
from exercise_code.networks.layer import *
from .gradient_check import eval_numerical_gradient_array


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


class AffineForwardTest(UnitTest):
    def __init__(self):
        self.error = 0.0

    def test(self):
        num_inputs = 2
        input_shape = (4, 5, 6)
        output_dim = 3

        input_size = num_inputs * np.prod(input_shape)
        weight_size = output_dim * np.prod(input_shape)

        x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
        w = np.linspace(-0.2, 0.3,
                        num=weight_size).reshape(np.prod(input_shape), output_dim)
        b = np.linspace(-0.3, 0.1, num=output_dim)

        out, _ = affine_forward(x, w, b)
        correct_out = np.array([[1.49834967, 1.70660132, 1.91485297],
                                [3.25553199, 3.5141327, 3.77273342]])

        self.error = rel_error(out, correct_out)

        return self.error < 1e-7

    def define_failure_message(self):
        return "Affine forward incorrect. Expected: < 1e-7 Evaluated: " + str(self.error)


class AffineBackwardTestDx(UnitTest):
    def __init__(self):
        self.error = 0.0

    def test(self):
        n = 10
        x = np.random.randn(n, 2, 3)
        w = np.random.randn(6, 5)
        b = np.random.randn(5)
        dout = np.random.randn(n, 5)

        dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)

        _, cache = affine_forward(x, w, b)
        dx, dw, db = affine_backward(dout, cache)

        self.error = rel_error(dx_num, dx)

        return self.error < 1e-7

    def define_failure_message(self):
        return "Affine backward wrt x incorrect. Expected: < 1e-7 Evaluated: " + str(self.error)


class AffineBackwardTestDw(UnitTest):
    def __init__(self):
        self.error = 0.0

    def test(self):
        n = 10
        x = np.random.randn(n, 2, 3)
        w = np.random.randn(6, 5)
        b = np.random.randn(5)
        dout = np.random.randn(n, 5)

        dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout) / n

        _, cache = affine_forward(x, w, b)
        dx, dw, db = affine_backward(dout, cache)

        self.error = rel_error(dw_num, dw)

        return self.error < 1e-7

    def define_failure_message(self):
        return "Affine backward wrt w incorrect. Expected: < 1e-7 Evaluated: " + str(self.error)


class AffineBackwardTestDb(UnitTest):
    def __init__(self):
        self.error = 0.0

    def test(self):
        n = 10
        x = np.random.randn(n, 2, 3)
        w = np.random.randn(6, 5)
        b = np.random.randn(5)
        dout = np.random.randn(n, 5)

        db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout) / n

        _, cache = affine_forward(x, w, b)
        dx, dw, db = affine_backward(dout, cache)

        self.error = rel_error(db_num, db)

        return self.error < 1e-7

    def define_failure_message(self):
        return "Affine backward wrt b incorrect. Expected: < 1e-7 Evaluated: " + str(self.error)


class AffineLayerTest(CompositeTest):
    def define_tests(self):
        return [
            AffineForwardTest(),
            AffineBackwardTestDx(),
            AffineBackwardTestDw(),
            AffineBackwardTestDb()
        ]

    def define_success_message(self):
        return "Congratulations you have passed all the unit tests!!!"

    def define_failure_message(self):
        return "Test cases are still failing!"


class SigmoidForwardTest(UnitTest):
    def __init__(self):
        self.error = 0.0

    def test(self):
        x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

        out, _ = sigmoid_forward(x)
        correct_out = np.array([[0.37754067, 0.39913012, 0.42111892, 0.44342513],
                                [0.46596182, 0.48863832, 0.51136168, 0.53403818],
                                [0.55657487, 0.57888108, 0.60086988, 0.62245933]])

        self.error = rel_error(out, correct_out)

        return self.error < 1e-6

    def define_failure_message(self):
        return "Sigmoid forward incorrect. Expected: < 1e-6 Evaluated: " + str(self.error)


class SigmoidBackwardTest(UnitTest):
    def __init__(self):
        self.error = 0.0

    def test(self):
        x = np.random.randn(10, 10)
        dout = np.random.randn(*x.shape)

        dx_num = eval_numerical_gradient_array(
            lambda x: sigmoid_forward(x)[0], x, dout)

        _, cache = sigmoid_forward(x)
        dx = sigmoid_backward(dout, cache)

        self.error = rel_error(dx_num, dx)

        return self.error < 1e-8

    def define_failure_message(self):
        return "Sigmoid backward incorrect. Expected: < 1e-8 Evaluated: " + str(self.error)


class SigmoidTest(CompositeTest):
    def define_tests(self):
        return [
            SigmoidForwardTest(),
            SigmoidBackwardTest()
        ]

    def define_success_message(self):
        return "Congratulations you have passed all the unit tests!!!"

    def define_failure_message(self):
        return "Test cases are still failing!"
