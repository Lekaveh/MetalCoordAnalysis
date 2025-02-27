import pytest
from metalCoord.analysis.classes import Class, Classificator


@pytest.fixture
def ideal_classes():
    """
    Create and return an instance of the Class.

    Returns:
        Class: An instance of the Class.
    """
    return Class()


@pytest.fixture
def classificator():
    """
    Create and return an instance of the Classificator.

    Returns:
        Classificator: An instance of the Classificator.
    """
    return Classificator()


def test_classificator_thr(classificator):
    """
    Test the initialization of the Classificator class.

    This test checks if the '_thr' attribute of the Classificator instance
    is correctly initialized to the value 0.3.
    """
    assert classificator.threshold == 0.3


def test_classificator_initialization(classificator):
    """
    Test the initialization of the Classificator class.

    This test ensures that an instance of the Classificator class can be created
    and verifies that the created instance is indeed of type Classificator.
    """

    assert isinstance(classificator, Classificator)


def test_ideal_classes_initialization(ideal_classes):
    """
    Test the initialization of the Class object.

    This test ensures that an instance of the Class object can be created
    and verifies that the created instance is indeed of type Class.
    """

    assert isinstance(ideal_classes, Class)


def test_class_number(ideal_classes):
    """
    Test the coordination number of the Class object.

    This test checks if the coordination number of the Class object is correctly
    initialized to the value 0.
    """

    assert len(ideal_classes.get_ideal_classes()) == 54


def test_zero_coordination(ideal_classes):
    """
    Test the coordination number of the Class object.

    This test checks if the coordination number of the Class object is correctly
    initialized to the value 0.
    """

    assert len(ideal_classes.get_ideal_classes_by_coordination(0)) == 0


def test_one_coordination(ideal_classes):
    """
    Test the coordination number of the Class object.

    This test checks if the coordination number of the Class object is correctly
    initialized to the value 0.
    """

    assert len(ideal_classes.get_ideal_classes_by_coordination(1)) == 0


def test_two_coordination(ideal_classes):
    """
    Test the coordination number of the Class object.

    This test checks if the coordination number of the Class object is correctly
    initialized to the value 0.
    """

    assert len(ideal_classes.get_ideal_classes_by_coordination(2)) == 2
