"""Tests for __len__() methods"""

from .base_tests import UnitTest, MethodTest


class LenTestInt(UnitTest):
    """Test whether __len__() method of an object returns type int"""
    def __init__(self, object_):
        self.object = object_

    def test(self):
        return isinstance(len(self.object), int)

    def define_failure_message(self):
        received_type = str(type(len(self.object)))
        return "Length is not of type int, got type %s." % received_type


class LenTestCorrect(UnitTest):
    """Test whether __len__() method of an object returns correct value"""
    def __init__(self, object_, len_):
        self.object = object_
        self.ref_len = len_

    def test(self):
        return len(self.object) == self.ref_len

    def define_failure_message(self):
        return "Length is incorrect (expected %d, got %d)." \
               % (self.ref_len, len(self.object))


class LenTest(MethodTest):
    """Test whether __len__() method of an object is correctly implemented"""
    def define_tests(self, object_, len_):
        return [LenTestInt(object_), LenTestCorrect(object_, len_)]

    def define_method_name(self):
        return "__len__"
