from .base_tests import UnitTest, CompositeTest, MethodTest, test_results_to_score
import numpy as np
import math
# from exercise_code.networks.loss import *

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """
    grad = np.zeros_like(x)
    oldval = x
    x = oldval + h # increment by h
    fxph = f(x) # evalute f(x + h)
    x = oldval - h
    fxmh = f(x) # evaluate f(x - h)
    x = oldval # restore

    # compute the partial derivative with centered formula
    grad = (fxph - fxmh) / (2 * h) # the slope
    if verbose:
        print(grad)

    return grad

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


class L1ForwardTest(UnitTest):
    def __init__(self, loss):
        self.value = loss.forward(np.array([5.,2.]), np.array([3.,4.]))
    
    def test(self):
        return (self.value == np.array([2.,2.])).all()

    def define_failure_message(self):
        return "L1 forward incorrect. Expected: [2.,2.] Evaluated: " + str(self.value)

class MSEForwardTest(UnitTest):
    def __init__(self, loss):
        self.value = loss.forward(np.array([7.,2.]), np.array([3.,4.]))
    
    def test(self):
        return (self.value == np.array([16.,4.])).all()

    def define_failure_message(self):
        return "MSE forward incorrect. Expected: [16.,4.] Evaluated: " + str(self.value)

class BCEForwardTest(UnitTest):
    def __init__(self, loss):
        self.value = loss.forward(np.array([0.7,0.2]), np.array([3.,4.]))
        self.truth = np.array([])
    def test(self):
        v1 = -3.0 * np.log(0.7) - (1-3.0) * np.log(1-0.7)
        v2 = -4.0 * np.log(0.2) - (1-4.0) * np.log(1-0.2)
        self.truth = np.array([v1, v2])
        
        return (rel_error(self.truth,self.value) < 1e-6)


    def define_failure_message(self):
        return "BCE forward incorrect. Expected: "+str(self.truth) +" Evaluated: " + str(self.value)        

class L1BackwardTestNormal(UnitTest):
    def __init__(self, loss):
        self.value = loss.backward(np.array([5.,2.]), np.array([3.,4.]))   
        self.loss = loss
        self.error = 0.0

    def test(self):
        f = lambda y: self.loss(y, np.array([3.,4.]))[0]
        num_grad = eval_numerical_gradient(f,np.array([5.,2.]), verbose = False)
        self.error = rel_error(num_grad,self.value)
        return self.error < 1e-8

    def define_failure_message(self):
        return "L1 backward incorrect. Expected: < 1e-8 Evaluated: " + str(self.error)

class L1BackwardTestZero(UnitTest):
    def __init__(self, loss):
        self.value = loss.backward(np.array([1.,1.]), np.array([1.,1.]))   

    def test(self):
        return (self.value == np.array([0.,0.])).all()

    def define_failure_message(self):
        return "L1 backward at 0 incorrect. Expected: [0.,0.] Evaluated: " + str(self.value)

class MSEBackwardTest(UnitTest):
    def __init__(self, loss):
        self.value = loss.backward(np.array([5.,2.]), np.array([3.,4.]))   
        self.loss = loss
        self.error = 0.0

    def test(self):
        f = lambda y: self.loss(y, np.array([3.,4.]))[0]
        num_grad = eval_numerical_gradient(f,np.array([5.,2.]), verbose = False)
        self.error = rel_error(num_grad,self.value)
        return self.error < 1e-8

    def define_failure_message(self):
        return "MSE backward incorrect. Expected: < 1e-8 Evaluated: " + str(self.error)

# class L1ForwardTest(MethodTest): 

class L1BackwardTest(MethodTest):
    def define_tests(self, loss):
        return [ L1BackwardTestNormal(loss), L1BackwardTestZero(loss)]

    def define_method_name(self):
        return "L1.backward"

    
class BCEBackwardTest(UnitTest):
    def __init__(self, loss):
        self.value = loss.backward(np.array([0.5,0.3]), np.array([2.,4.]))   
        self.loss = loss
        self.error = 0.0

    def test(self):
        f = lambda y: self.loss(y, np.array([2.,4.]))[0]
        num_grad = eval_numerical_gradient(f,np.array([0.5,0.3]), verbose = False)
        self.error = rel_error(num_grad,self.value)
        return self.error < 1e-8

    def define_failure_message(self):
        return "BCE backward incorrect. Expected: < 1e-8 Evaluated: " + str(self.error)

class L1Test(CompositeTest):
    def define_tests(self, loss):
        return [
            L1ForwardTest(loss),
            L1BackwardTestZero(loss),
            L1BackwardTestNormal(loss),
        ]
    def define_success_message(self):
        return "Congratulations you have passed all the unit tests!!!"

    def define_failure_message(self):
        return "Test cases are still failing!"

class MSETest(CompositeTest):
    def define_tests(self, loss):
        return [
            MSEForwardTest(loss),
            MSEBackwardTest(loss)
        ]
    def define_success_message(self):
        return "Congratulations you have passed all the unit tests!!!"

    def define_failure_message(self):
        return "Test cases are still failing!"

class BCETest(CompositeTest):
    def define_tests(self, loss):
        return [
            BCEForwardTest(loss),
            BCEBackwardTest(loss)
        ]
    def define_success_message(self):
        return "Congratulations you have passed all the unit tests!!!"

    def define_failure_message(self):
        return "Test cases are still failing!"
