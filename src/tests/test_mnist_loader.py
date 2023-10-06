import numpy as np
import pytest
from mnist_loader import load_data_wrapper, vectorized_result

# Define a fixture to load data_wrapper for multiple tests
@pytest.fixture
def data_wrapper():
    return load_data_wrapper()

# Test the load_data_wrapper function
def test_load_data_wrapper(data_wrapper):
    training_data, validation_data, test_data = data_wrapper
    # Check if the data is not empty
    assert training_data
    assert validation_data
    assert test_data

    # Check the dimensions of the loaded data
    assert len(training_data[0][0]) == 784
    assert len(training_data[0][1]) == 10

    # Ensure that the data contains the correct number of samples
    assert len(training_data) == 50000
    assert len(validation_data) == 10000
    assert len(test_data) == 10000

# Test the vectorized_result function
def test_vectorized_result():
    # Test the function for different values of j
    for j in range(10):
        result = vectorized_result(j)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 1)
        assert result[j][0] == 1.0

