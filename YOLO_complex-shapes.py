import torch
from ultralytics import YOLO
import os
import random
from PIL import Image as PILImage
from IPython.display import display
import numpy as np
import cv2

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

model = YOLO("yolov8n.pt")
print("YOLOv8 loaded successfully")


from roboflow import Roboflow
rf = Roboflow(api_key="QQoeAj8Mx9NUigim0MXR")
project = rf.workspace("divyanshi-sctp1").project("fujiko-5hpzt")
version = project.version(3)
dataset = version.download("yolov8")


model = YOLO("yolov8n.pt")

model.train(
    data=r"E:\DS project\YOLO_DORAEMON_NEW\Fujiko-3\data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device='cpu'  
)


#Load the trained model
weights_path = r"E:\DS project\YOLO_DORAEMON_NEW\runs\detect\train7\weights\best.pt"
model = YOLO(weights_path)

#Path to test images
test_dir = r"E:\DS project\YOLO_DORAEMON_NEW\Fujiko-3\test\images"

#Pick a random image
all_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
random_image = random.choice(all_images)
image_path = os.path.join(test_dir, random_image)
print(f"Running inference on: {random_image}")

#Run prediction
results = model.predict(source=image_path, conf=0.25, save=False)

#Get the original image (RGB) from YOLO results
orig_img_rgb = results[0].orig_img.copy()  # NumPy array in RGB

# Overlay bounding boxes and class labels
annotated_img = results[0].plot()  # returns RGB array (but may be interpreted as BGR)

#BGR to RGB to fix color inversion
rgb_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

#Convert to PIL Image for Jupyter display
display_img = PILImage.fromarray(rgb_img)

# Image display
display(display_img)


#Path to the external image you want to test
image_path = r"E:\DS project\YOLO_DORAEMON_NEW\download (2).png"

#Run prediction
results = model.predict(source=image_path, conf=0.25, save=False)

#Get annotated image and fix color
# results[0].plot() returns an image array (usually RGB, but sometimes interpreted as BGR)
annotated_img = results[0].plot()

# Convert BGR to RGB to ensure correct display
rgb_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

#Display image in Jupyter
display_img = PILImage.fromarray(rgb_img)
display(display_img)


 #  $$$$ ____TESTING A VIDEO_____ $$$$ 

from collections import Counter

# Load your trained model
weights_path = r"E:\DS project\YOLO_DORAEMON_NEW\runs\detect\train7\weights\best.pt"
model = YOLO(weights_path)

# Path to your input video
video_path = r"E:\DS project\YOLO_DORAEMON_NEW\test_video.mp4"

# Open video file
cap = cv2.VideoCapture(video_path)

# Get video details
fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video writer to save the output
output_path = r"E:\DS project\YOLO_DORAEMON_NEW\output_detected_video.mp4"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Initialize global counter
total_counts = Counter()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO prediction on each frame
    results = model.predict(frame, conf=0.25, verbose=False)

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    # Count detected objects for this frame
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    class_names = [results[0].names[i] for i in class_ids]
    total_counts.update(class_names)

    # Show live window (optional)
    cv2.imshow("YOLO Video Detection", annotated_frame)

    # Write annotated frame to output video
    out.write(annotated_frame)

    # Press 'q' to stop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Print summary of total objects detected in video
print("\n Total Object Counts in Video:")
for label, count in total_counts.items():
    print(f"{label}: {count}")

print(f"\n Output video saved at: {output_path}")
           