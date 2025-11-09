import numpy as np
import pytest
from tqdm import tqdm
from metalCoord.analysis.classes import Class
from metalCoord.correspondense.procrustes import fit


@pytest.fixture
def ideal_classes():
    """
    Create and return an instance of the Class.

    Returns:
        Class: An instance of the Class.
    """
    return Class()


def permute(coord: np.ndarray) -> np.ndarray:
    """
    Permute the rows of the input array.

    This function permutes the rows of the input array and returns the
    permuted array.

    Args:
        coord (np.ndarray): A 2D numpy array.

    Returns:
        np.ndarray: A 2D numpy array with permuted rows.
    """
    perm = np.random.permutation(coord.shape[0] - 1) + 1
    perm = np.insert(perm, 0, 0)
    return coord[perm]


def test_non_sandwiches(ideal_classes: Class):
    """
    Test function to validate the fit of ideal classes that do not contain the term "sandwich".

    Args:
        ideal_classes (IdealClasses): An instance of IdealClasses containing the ideal classes to be tested.

    This function filters out any ideal classes that contain the term "sandwich" and then validates the fit of the remaining ideal classes.
    """

    ideals = ideal_classes.get_ideal_classes()
    ideals = [ideal for ideal in ideals if "sandwich" not in ideal and ideal !=
              "penta-trigonal-planar" and ideal_classes.get_coordination(ideal) <= 12]
    validate_ideal_class_fit(ideal_classes, ideals)


def test_sandwiches(ideal_classes: Class):
    """
    Test function to validate the fit of ideal classes containing the term "sandwich".

    Args:
        ideal_classes (Class): An instance of the Class containing ideal classes.

    Returns:
        None
    """

    ideals = ideal_classes.get_ideal_classes()
    ideals = [ideal for ideal in ideals if "sandwich" in ideal or ideal ==
              "penta-trigonal-planar"]
    validate_ideal_class_fit(ideal_classes, ideals)


def validate_ideal_class_fit(ideal_classes, ideals):
    """
    Validates the fit of ideal classes against given ideals.
    This function iterates over a list of ideals, retrieves their coordinates
    from the ideal_classes object, and checks if the fit of the coordinates
    to themselves and their permuted versions is within a specified tolerance.
    Parameters:
    ideal_classes (object): An object that provides a method get_coordinates(ideal)
                            to retrieve the coordinates of a given ideal.
    ideals (list): A list of ideals to be tested.
    Raises:
    AssertionError: If the fit of the coordinates to themselves or their permuted
                    versions is not within the specified tolerance.
    """
    for ideal in (pbar := tqdm(ideals, desc="Testing Ideal Classes")):
        pbar.set_postfix(ideal=ideal)

        coord = ideal_classes.get_coordinates(ideal)
        if len(coord) > 12:
            continue

        assert np.allclose([fit(coord, coord)[0]], [0], atol=1e-5)

        permuted_coord = permute(coord)
        assert np.allclose([fit(coord, permuted_coord)[0]], [0], atol=1e-5)
