import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(title, image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,8))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

def detect_shapes(image_path, target_size=(800, 800)):
    # Load and resize image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image. Please check the file path.")
        return
