import pytest
import numpy as np
import gzip
import pickle
from mnist_loader import load_data, load_data_wrapper, vectorized_result


@pytest.fixture
def mnist_data():
    with gzip.open("src/training_data/mnist_expanded.pkl.gz", "rb") as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    return training_data, validation_data, test_data

def test_load_data(mnist_data):
    training_data, validation_data, test_data = mnist_data
    loaded_training_data, loaded_validation_data, loaded_test_data = load_data()

    assert np.array_equal(training_data, loaded_training_data)
    assert np.array_equal(validation_data, loaded_validation_data)
    assert np.array_equal(test_data, loaded_test_data)

def test_vectorized_result():
    # Test for vectorized_result function
    result = vectorized_result(3)
    expected_result = np.array([[0], [0], [0], [1], [0], [0], [0], [0], [0], [0]])
    assert np.array_equal(result, expected_result)

def test_load_data_wrapper(mnist_data):
    training_data, _, _ = mnist_data
    loaded_training_data, _, _ = load_data_wrapper()

    assert len(loaded_training_data) == len(training_data)
    assert len(loaded_training_data[0][0]) == 784
    assert len(loaded_training_data[0][1]) == 10

