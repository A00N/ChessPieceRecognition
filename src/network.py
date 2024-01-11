import random
import pickle
import numpy as np
 
random.seed(125)
np.random.seed(125)
 
class CrossEntropyCost(object):
    """
    Cross-entropy cost function for neural network training.
 
    Attributes:
        None
    """
 
    @staticmethod
    def fn(a, y):
        """
        Calculate the cost associated with the predicted 'a' and actual 'y' values.
 
        Args:
            a (numpy.ndarray): The predicted output.
            y (numpy.ndarray): The actual output.
 
        Returns:
            float: The cost.
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
 
    @staticmethod
    def delta(z, a, y):
        """
        Calculate the error delta for backpropagation.
 
        Args:
            z (numpy.ndarray): The input to the activation function.
            a (numpy.ndarray): The activation output.
            y (numpy.ndarray): The actual output.
 
        Returns:
            numpy.ndarray: The error delta.
        """
        return (a-y)
 
 
class Network(object):
    """
    A feedforward neural network.
 
    Attributes:
        sizes (list): The number of neurons in each layer of the network.
        num_layers (int): The number of layers in the network.
        biases (list): List of bias vectors for each layer.
        weights (list): List of weight matrices for each layer.
        cost (CrossEntropyCost): The cost function used for training.
    """
 
    def vectorized_result(j):
        """
        Create a one-hot encoded vector for a digit.
 
        Args:
            j (int): The digit to be encoded.
 
        Returns:
            numpy.ndarray: The one-hot encoded vector.
        """
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e
 
    def sigmoid(self, z):
        """
        Compute the sigmoid function for a given input 'z'.
 
        Args:
            z (numpy.ndarray): The input to the sigmoid function.
 
        Returns:
            numpy.ndarray: The sigmoid of 'z'.
        """
        return 1.0 / (1.0 + np.exp(-z))
 
    def sigmoid_prime(self, z):
        """
        Compute the derivative of the sigmoid function for a given input 'z'.
 
        Args:
            z (numpy.ndarray): The input to the sigmoid function.
 
        Returns:
            numpy.ndarray: The derivative of the sigmoid of 'z'.
        """
        return self.sigmoid(z) * (1 - self.sigmoid(z))
 
    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        Initialize the network with layer sizes and a cost function (CrossEntropyCost).
 
        Args:
            sizes (list): The number of neurons in each layer.
            cost (CrossEntropyCost): The cost function used for training.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weight_initializer()
        self.cost=cost
 
    def weight_initializer(self):
        """
        Initialize network weights and biases.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
 
 
    def feedforward(self, a):
        """
        Feed input 'a' forward through the network and return the output.
 
        Args:
            a (numpy.ndarray): The input data.
 
        Returns:
            numpy.ndarray: The network's output.
        """
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a)+b)
        return a
 
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda,
            evaluation_data=None,
            monitor_evaluation_accuracy=False,
            monitor_training_accuracy=False):
        """
        Stochastic Gradient Descent (SGD) training method.
 
        Args:
            training_data (list): The training data.
            epochs (int): The number of training epochs.
            mini_batch_size (int): The size of mini-batches.
            eta (float): The learning rate.
            lmbda (float): The L2 regularization parameter.
            evaluation_data (list): Data for evaluation.
            monitor_evaluation_accuracy (bool): Whether to monitor evaluation accuracy.
            monitor_training_accuracy (bool): Whether to monitor training accuracy.
 
        Returns:
            Tuple: Evaluation cost, evaluation accuracy, training cost, and training accuracy.
        """
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" % j)
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data))
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy
 
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        Update network weights and biases based on a mini-batch of training data.
 
        Args:
            mini_batch (list): The mini-batch of training data.
            eta (float): The learning rate.
            lmbda (float): The L2 regularization parameter.
            n (int): The total number of training examples. 
 
        Returns:
            None
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b, nabla_w = self.backprop(mini_batch)
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
 
    def backprop(self,minibatch):
   # Extract 'x' values
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
 
        xs = [x for x, y in minibatch]
        x_matrix = np.hstack(xs)
        ys = [y for x,y in minibatch]
        y_matrix = np.hstack(ys)
        activation = x_matrix
        activations = [x_matrix]
 
        zs = [] # list to store all the z vectors, layer by layer
 
        for b, w in zip(self.biases, self.weights):
                    z = np.dot(w, activation)+b
                    zs.append(z)
                    activation = self.sigmoid(z)
                    activations.append(activation)
 
        delta = (self.cost).delta(zs[-1],activations[-1], y_matrix) 
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
 
        for l in range(2, self.num_layers):
                    z = zs[-l]
                    sp = self.sigmoid_prime(z)
                    delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
                    nabla_b[-l] = delta
                    nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
 
        nabla_b_summed = [np.sum(nb, axis=1, keepdims=True) for nb in nabla_b]
        return (nabla_b_summed, nabla_w)
 
    def accuracy(self, data, convert=False):
        """
        Calculate the accuracy of the network's predictions.
 
        Args:
            data (list): Data for evaluation.
            convert (bool): Whether to convert the output for comparison.
 
        Returns:
            int: The number of correct predictions.
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
 
    def total_cost(self, data, lmbda, convert=False):
        """
        Calculate the total cost of the network on a given dataset.
 
        Args:
            data (list): Data for evaluation.
            lmbda (float): The L2 regularization parameter.
            convert (bool): Whether to convert the output for comparison.
 
        Returns:
            float: The total cost.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = self.vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost
 
    def save(self, filename):
        """
        Save the network to a file.
 
        Args:
            filename (str): The filename for saving the network.
 
        Returns:
            None
        """
        data = {
            "sizes": self.sizes,
            "biases": [b.tolist() for b in self.biases],
            "weights": [w.tolist() for w in self.weights]
        }
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
 
    def load(self, filename):
        """
        Load the network from a saved file.
 
        Args:
            filename (str): The filename from which to load the network.
 
        Returns:
            None
        """
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        self.sizes = data["sizes"]
        self.biases = [np.array(b) for b in data["biases"]]
        self.weights = [np.array(w) for w in data["weights"]]
        self.num_layers = len(self.sizes)