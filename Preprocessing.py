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

    img_resized = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # --- Preprocessing ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    edges = cv2.Canny(clean, 100, 200)
    #combined = cv2.bitwise_or(clean, edges)
    
    # --- Find contours ---
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_counts = {
        "Triangle": 0,
        "Square": 0,
        "Rectangle": 0,
        "Quadrilateral": 0,
        "Pentagon": 0,
        "Hexagon": 0,
        "Circle": 0,
        "Polygon": 0
    }

