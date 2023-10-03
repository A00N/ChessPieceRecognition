import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import network
import cv2

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")
        self.size = 280
        self.brush = 10

        self.canvas = tk.Canvas(root, width=self.size, height=self.size, bg="black")  # Set the canvas background to black
        self.canvas.pack()

        self.label = tk.Label(root, text="", fg="black")  # Set label text color to white
        self.label.pack()

        self.predict_button = tk.Button(root, text="PREDICT", command=self.predict_digit)
        self.predict_button.pack()

        self.clear_button = tk.Button(root, text="CLEAR", command=self.clear_canvas)
        self.clear_button.pack()

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)

        # Initialize drawing
        self.drawing = False
        self.image = Image.new("L", (self.size, self.size), "black")  # Set the image background to black
        self.draw = ImageDraw.Draw(self.image)

        # Load trained network
        self.net = network.Network([28 * 28, 30, 10])
        self.net.load('src/trained_networks/network_25_09_23.pkl')

    def start_drawing(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y


    """
    Function which handles the drawing to canvas.
    """
    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            # Draw a small oval in white
            self.canvas.create_oval(
                x - self.brush, y - self.brush, x + self.brush, y + self.brush,
                fill="white", outline="white"
            )
            # Draw on the image as well
            self.draw.ellipse([x - self.brush, y - self.brush, x + self.brush, y + self.brush], fill="white", outline="white")


    """
    Cleares canvas.
    """
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.size, self.size), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text="")
    
    
    
    def predict_digit(self):
        # Resize the drawn image to 28x28 pixels
        resized_image = self.image.resize((28, 28), Image.BICUBIC)
        # Convert the resized image to a numpy array
        resized_image_array = np.array(resized_image)
        # Calculate the center of mass (centroid) of the digit image
        M = cv2.moments(resized_image_array)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        # Calculate the offset needed to move the center of mass to the center of the canvas
        offset_x = 14 - cx  # 14 is the desired center in a 28x28 canvas
        offset_y = 14 - cy
        # Create a translation matrix
        translation_matrix = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        # Apply the offset to the digit image
        centered_image = cv2.warpAffine(resized_image_array, translation_matrix, (28, 28))
        centered_image = centered_image.reshape((28 * 28, 1)) / 255.0
        predicted_digit = np.argmax(self.net.feedforward(centered_image))
        self.label.config(text=f"Predicted Digit: {predicted_digit}")



if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
