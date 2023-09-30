import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import network

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")
        self.size = 112

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

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            # Draw a small oval in white
            self.canvas.create_oval(
                x - 5, y - 5, x + 5, y + 5,
                fill="white", outline="white"
            )
            # Draw on the image as well
            self.draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill="white", outline="white")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (112, 112), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text="")

    def predict_digit(self):
        # Resize and normalize the drawn image
        resized_image = self.image.resize((28, 28), Image.BICUBIC)
        centered_image = Image.new("L", (28, 28), "black")
        centered_image.paste(resized_image, (0, 0))
        centered_image = np.array(centered_image)
        centered_image = centered_image.reshape((28 * 28, 1)) / 255.0

        predicted_digit = np.argmax(self.net.feedforward(centered_image))
        self.label.config(text=f"Predicted Digit: {predicted_digit}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
