import datetime 
import mnist_loader 
import network 
import time 
 
 
 
 
time_1 = time.time() 
# Load data and create the network 
training_data, validation_data, test_data = mnist_loader.load_data_wrapper() 
print("Data loaded") 
 
 
net = network.Network([784, 30, 10]) 
 
# Hyperparameters 
epochs = 50                # Number of training epochs 
mini_batch_size = 50        # Mini-batch size for stochastic gradient descent 
learning_rate = 0.1         # Learning rate for weight updates 
lmbda = 5                   # L2 regularization strength 

# Training the network 
net.SGD(training_data, epochs, mini_batch_size, learning_rate, lmbda,  
         evaluation_data=validation_data, 
         monitor_training_accuracy=True, 
         monitor_evaluation_accuracy=True) 
 
print("Training completed") 
 
time_2 = time.time() 
 
print(time_2-time_1) 
 
# Get the current date 
current_date = datetime.datetime.now().strftime("%d_%m_%y") 
 
# Save the trained network with the current date 
filename = f"src/trained_networks/network_{current_date}.pkl" 
net.save(filename) 
 
print(f"Trained network saved as {filename}")