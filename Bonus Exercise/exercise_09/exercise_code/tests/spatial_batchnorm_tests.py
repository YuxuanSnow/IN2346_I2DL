import numpy as  np

from .base_tests import UnitTest, CompositeTest
from .. import layers
from .gradient_check import (
    eval_numerical_gradient_array,
    eval_numerical_gradient,
    rel_error,
)

class SpatialBatchnormForwardTest(UnitTest):
    def __init__(self, shape, mean, scale, beta, gamma, mode, test_name):
        np.random.seed(0)
        
        self.x = scale * np.random.randn(*shape) + mean
        self.beta = beta
        self.gamma = gamma
        self.bn_param = {'mode' : mode}
        self.test_name = test_name
        
        # Check the test-time forward pass by running the training-time
        # forward pass many times to warm up the running averages, and then
        # checking the means and variances of activations after a test-time
        # forward pass.
        if mode == 'test':
            self.bn_param['mode'] = 'train'
            for t in range(50):
                x = scale * np.random.randn(*shape) + mean
                layers.spatial_batchnorm_forward(x, gamma, beta, self.bn_param)
            self.bn_param['mode'] = 'test'

            
    def test(self):
        out, _ = layers.spatial_batchnorm_forward(
            self.x, self.gamma, self.beta, self.bn_param)
        out_mean = out.mean(axis=(0, 2, 3))
        out_std = out.std(axis=(0, 2, 3))
        
        atol = 1e-5 if self.bn_param['mode'] == 'train' else 0.15
        return np.all(np.isclose(self.beta, out_mean, atol=atol)) and \
               np.all(np.isclose(self.gamma, out_std, atol=atol))
    
    def define_failure_message(self):
        return '%s failed.' % self.test_name
    
    def define_success_message(self):
        return '%s passed.' % self.test_name
    

class SpatialBatchnormBackwardTest(UnitTest):
    def __init__(self, shape, mean, scale, beta, gamma, mode):
        np.random.seed(0)
        
        self.x = scale * np.random.randn(*shape) + mean
        self.dout = np.random.randn(*shape)
        
        self.beta = beta
        self.gamma = gamma
        self.bn_param = {'mode' : mode}
        
    def test(self):
        fx = lambda x: layers.spatial_batchnorm_forward(
            x, self.gamma, self.beta, self.bn_param)[0]
        fg = lambda a: layers.spatial_batchnorm_forward(
            self.x, a, self.beta, self.bn_param)[0]
        fb = lambda b: layers.spatial_batchnorm_forward(
            self.x, self.gamma, b, self.bn_param)[0]

        dx_num = eval_numerical_gradient_array(fx, self.x, self.dout)
        da_num = eval_numerical_gradient_array(fg, self.gamma, self.dout)
        db_num = eval_numerical_gradient_array(fb, self.beta, self.dout)

        _, cache = layers.spatial_batchnorm_forward(
            self.x, self.gamma, self.beta, self.bn_param)
        dx, dgamma, dbeta = layers.spatial_batchnorm_backward(
            self.dout, cache)
        
        return np.isclose(rel_error(dx_num, dx), 0, atol=1e-6) and \
               np.isclose(rel_error(da_num, dgamma), 0, atol=1e-6) and \
               np.isclose(rel_error(db_num, dbeta), 0, atol=1e-6)

    
class SpatialBatchnormForwardTests(CompositeTest):
    def define_tests(self):
        return [
            SpatialBatchnormForwardTest(shape=(2, 3, 4, 5), 
                                        mean=10,
                                        scale=4,
                                        beta=np.zeros(3),
                                        gamma=np.ones(3),
                                        mode='train',
                                        test_name='SpatialBatchnormForwardTest with trivial beta and gamma (train)'),
            SpatialBatchnormForwardTest(shape=(2, 3, 4, 5), 
                                        mean=10,
                                        scale=4,
                                        beta=np.array([6, 7, 8]),
                                        gamma=np.array([3, 4, 5]),
                                        mode='train',
                                        test_name='SpatialBatchnormForwardTest with nontrivial beta and gamma (train)'),
            SpatialBatchnormForwardTest(shape=(10, 4, 11, 12), 
                                        mean=13,
                                        scale=2.3,
                                        beta=np.zeros(4),
                                        gamma=np.ones(4),
                                        mode='test',
                                        test_name='SpatialBatchnormForwardTest with trivial beta and gamma (test)')
        ]

    def define_failure_message(self):
        return "Some tests failed for your spatial batchnorm implementation."

    def define_success_message(self):
        return "All tests passed for your spatial batchnorm implementation."
    

def test_spatial_batchnorm_forward():
    SpatialBatchnormForwardTests()()
    
def test_spatial_batchnorm_backward():
    SpatialBatchnormBackwardTest(shape=(2, 3, 4, 5),
                                 mean=12,
                                 scale=5,
                                 beta=np.random.randn(3),
                                 gamma=np.random.randn(3),
                                 mode='train')()

        