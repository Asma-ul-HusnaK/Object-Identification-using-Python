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
    # --- Shape Detection ---
    for cnt in contours:
        area = cv2.contourArea(cnt)

        perimeter = cv2.arcLength(cnt, True)

        approx = cv2.approxPolyDP(cnt, 0.02*perimeter, True)
        #print(approx)
        vertices = len(approx)
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h

        shape_name = "Unknown"
        circularity = (4 * np.pi * area) / (perimeter * perimeter)

        if circularity > 0.83 and vertices > 6:
            shape_name = "Circle"
        elif circularity > 0.6 and vertices > 6:
            shape_name = "Polygon"
        else:
            if vertices == 3:
                shape_name = "Triangle"
            elif vertices == 4:
                def angle(pt1, pt2, pt3):
                    v1 = pt1 - pt2
                    v2 = pt3 - pt2
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    return np.degrees(np.arccos(cos_angle))

                angles = []
                for i in range(4):
                    pt1 = approx[i][0]
                    pt2 = approx[(i + 1) % 4][0]
                    pt3 = approx[(i + 2) % 4][0]
                    angles.append(angle(pt1, pt2, pt3))

                if all(85 <= a <= 95 for a in angles):
                    if 0.95 <= aspect_ratio <= 1.05:
                        shape_name = "Square"
                    else:
                        shape_name = "Rectangle"
                else:
                    shape_name = "Quadrilateral"
            elif vertices == 5:
                shape_name = "Pentagon"
            elif vertices == 6:
                shape_name = "Hexagon"
            elif vertices > 6:
                shape_name = "Polygon"

        if shape_name in shape_counts:
            shape_counts[shape_name] += 1
        else:
            shape_counts["Polygon"] += 1

        margin = 10
        x1 = max(x - margin, 0)
        y1 = max(y - margin, 0)
        x2 = min(x + w + margin, img_resized.shape[1] - 1)
        y2 = min(y + h + margin, img_resized.shape[0] - 1)

        cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(img_resized, shape_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    print("\n Shape Counts Detected:")
    for shape, count in shape_counts.items():
        print(f"{shape}: {count}")
        
 #  Display image here inside the function
    show_image("Threshold_image", thresh)
    show_image("Detected Shapes", img_resized)


# --- Run the detection ---
if __name__ == "__main__":
    image_path = "E:\DS project\Git_push\download.jpeg"  # Replace with your image path
    detect_shapes(image_path)


