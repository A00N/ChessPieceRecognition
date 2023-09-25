from PIL import Image
import numpy as np
import network


def main(): 
    # Load trained network
    net = network.Network([784, 30, 10]) # use same net as in start.py
    net.load('src/trained_networks/network.pkl')  # Load selected network version

    # Load image
    image = Image.open('src/test_data/number.png').convert('L')  # Convert to grayscale
    image = np.array(image)  # Convert to NumPy array
    image = image.reshape((784, 1))  # Reshape to (784, 1)
    image = image / 255.0  # Normalize pixel values to [0, 1]

    # Use trained network to make prediction
    predicted_digit = np.argmax(net.feedforward(image))

    print("Predicted Digit:", predicted_digit)


if __name__ == "__main__":
    main()
