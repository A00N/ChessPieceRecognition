import pytest
import numpy as np
from network import Network


@pytest.fixture
def simple_network():
    # Create a simple network for testing
    return Network([2, 3, 2])


def test_feedforward(simple_network):
    # Test feedforward method
    activations = simple_network.feedforward(np.array([[1], [2]]))
    assert activations.shape == (2, 1)


def test_backpropagation(simple_network):
    # Test backpropagation method
    x = np.array([[1], [2]])
    y = np.array([[0], [1]])
    nabla_b, nabla_w = simple_network.backprop(x, y)
    assert len(nabla_b) == len(nabla_w) == len(simple_network.biases) == len(simple_network.weights)


def test_save_and_load(simple_network, tmp_path):
    # Test save and load methods
    filename = tmp_path / "test_network.pkl"
    simple_network.save(filename)
    loaded_network = Network([2, 3, 2])
    loaded_network.load(filename)
    assert simple_network.sizes == loaded_network.sizes
    assert len(simple_network.biases) == len(loaded_network.biases)
    assert len(simple_network.weights) == len(loaded_network.weights)


if __name__ == "__main__":
    pytest.main()