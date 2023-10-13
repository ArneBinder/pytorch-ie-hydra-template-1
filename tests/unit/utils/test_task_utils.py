import pytest

from src.utils.task_utils import get_required_arguments_of_class


class ExampleClass:
    def __init__(self, arg1, arg2, arg3="default_value"):
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3


class ClassWithoutInit:
    pass


def test_get_required_arguments_of_class_valid():
    required_args = get_required_arguments_of_class(ExampleClass)
    assert required_args == ["arg1", "arg2"]


def test_get_required_arguments_of_class_no_init_method():
    with pytest.raises(ValueError, match="The class does not have an __init__ method."):
        get_required_arguments_of_class(ClassWithoutInit)


def test_get_required_arguments_of_class_not_a_class():
    with pytest.raises(ValueError, match="Input must be a class type."):
        get_required_arguments_of_class("This is not a class")
