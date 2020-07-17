"""Abstract test classes"""

# pylint: disable=lost-exception

from abc import ABC, abstractmethod


class UnitTest(ABC):
    """
    Abstract class for a single test
    All subclasses have to overwrite test() and failure_message()
    Then the execution order is the following:
        1. test() method is executed
        2. if test() method returned False or threw an exception,
            print the failure message defined by failure_message()
        3.  return a tuple (tests_failed, total_tests)
    """

    def __call__(self):
        try:
            test_passed = self.test()
            if test_passed:
                print(self.define_success_message())
                return 0, 1  # 0 tests failed, 1 total test
            print(self.define_failure_message())
            return 1, 1  # 1 test failed, 1 total test
        except Exception as exception:
            print(self.define_exception_message(exception))
            return 1, 1  # 1 test failed, 1 total test

    @abstractmethod
    def test(self):
        """Run the test and return True if passed else False"""

    def define_failure_message(self):
        """Define the message that should be printed upon test failure"""
        return "%s failed." % type(self).__name__

    def define_success_message(self):
        """Define the message that should be printed upon test success"""
        return "%s passed." % type(self).__name__

    def define_exception_message(self, exception):
        """
        Define the message that should be printed if an exception occurs
        :param exception: exception that was thrown
        """
        return "%s failed due to exception: %s." \
               % (type(self).__name__, exception)


class CompositeTest(ABC):
    """
    Abstract class for a test consisting of multiple other tests
    All subclasses have to overwrite define_tests(), success_message(),
    and failure_message().
    Then the execution order is the following:
    1. run all tests
    2. if all tests passed, print success message
    3. if some tests failed, print failure message
         and how many tests passed vs total tests
    4. return a tuple (tests_failed, total_tests)
    """
    def __init__(self, *args, **kwargs):
        self.tests = self.define_tests(*args, **kwargs)

    @abstractmethod
    def define_tests(self, *args, **kwargs):
        """Define a list of all sub-tests that should be run"""

    def define_success_message(self):
        """Define message to be printed if all tests succeed"""
        return "All tests of %s passed." % type(self).__name__

    def define_failure_message(self):
        """Define message to be printed if some tests fail"""
        return "Some tests of %s failed." % type(self).__name__

    def __call__(self):
        tests_failed, tests_total = 0, 0
        for test in self.tests:
            new_fail, new_total = test()
            tests_failed += new_fail
            tests_total += new_total
        tests_passed = tests_total - tests_failed
        if tests_failed == 0:
            print(
                self.define_success_message(),
                "Tests passed: %d/%d" % (tests_passed, tests_total)
            )
        else:
            print(
                self.define_failure_message(),
                "Tests passed: %d/%d" % (tests_passed, tests_total)
            )
        return tests_failed, tests_total


class MethodTest(CompositeTest, ABC):
    """
    Abstract class to test methods using multiple tests
    Similar behaviour to CompositeTest, except that subclasses have to
    overwrite define_method_name instead of success_message and failure_message
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method_name = self.define_method_name()

    @abstractmethod
    def define_method_name(self):
        """Define name of the method to be tested"""

    def define_success_message(self):
        return "Method %s() correctly implemented." % self.method_name

    def define_failure_message(self):
        return "Some tests failed for method %s()." % self.method_name


class ClassTest(CompositeTest, ABC):
    """
    Abstract class to test classes using multiple tests
    Similar behaviour to CompositeTest, except that subclasses have to
    overwrite define_class_name instead of success_message and failure_message
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_name = self.define_class_name()

    @abstractmethod
    def define_class_name(self):
        """Define name of the class to be tested"""

    def define_success_message(self):
        return "Class %s correctly implemented." % self.class_name

    def define_failure_message(self):
        return "Some tests failed for class %s." % self.class_name


def test_results_to_score(test_results, verbose=True):
    """Calculate a score from 0-100 based on number of failed/total tests"""
    tests_failed, tests_total = test_results
    tests_passed = tests_total - tests_failed
    score = int(100 * tests_passed / tests_total)
    if verbose:
        print("Score: %d/100" % score)
    return score
