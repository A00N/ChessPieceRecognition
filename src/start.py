import mnist_loader
import network
import datetime

# Load data and create the network
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print("Data loaded")

net = network.Network([784, 128, 64, 10]) # input layer (28x28 pixels = 784 neurons), hidden layer, hidden layer , output layer (possible 10 numbers 0-9).

# Train the network
net.SGD(training_data, 10, 64, 1, test_data=test_data) # training_data, epochs, mini_batch_size, eta, test_data
print("Training completed")

# Get the current date
current_date = datetime.datetime.now().strftime("%d_%m_%y")

# Save the trained network with the current date
filename = f"src/trained_networks/network_{current_date}.pkl"
net.save(filename)

print(f"Trained network saved as {filename}")
