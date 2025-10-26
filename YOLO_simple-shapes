import torch
from ultralytics import YOLO

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

model = YOLO("yolov8n.pt")
print("YOLOv8 loaded successfully")

from roboflow import Roboflow
rf = Roboflow(api_key="QQoeAj8Mx9NUigim0MXR")
project = rf.workspace("divyanshi-sctp1").project("yolo_ds-project-qpbnj")
version = project.version(7)
dataset = version.download("yolov8")

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Train the model
results = model.train(
    data="E:\DS project\YOLO_SIMPLE_SHAPES\YOLO_DS.Project-7\data.yaml",  # path to your dataset YAML
    epochs=60,                 # number of epochs
    imgsz=640,                 # image size
    batch=16,                  # batch size
    project="runs/train",      # main folder to store runs
    name="yolov8_custom",      # subfolder name (will always overwrite)
    patience=10,               # Early stopping if no improvement
    exist_ok=True,             # overwrite if folder already exists
    save=True,                 # save final weights
)



from IPython.display import Image, display
import os
import random

# Loading the trained model
weights_path = r"E:\DS project\YOLO_SIMPLE_SHAPES\runs\train\yolov8_custom\weights\best.pt"
model = YOLO(weights_path)

# Testing the images
test_folder = r"E:\DS project\YOLO_SIMPLE_SHAPES\YOLO_DS.Project-7\test\images"

# chechng for a random image
all_images = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.png'))]
random_image = random.choice(all_images)
image_path = os.path.join(test_folder, random_image)

print(f"Running inference on: {image_path}")

# displaying bounding boxes with running predictions
results = model.predict(
    source=image_path,
    conf=0.5,
    save=True,       # saves annotated image
    imgsz=640,
    device='cpu'
)

# Annoted image path
annotated_dir = results[0].save_dir
annotated_image_path = os.path.join(annotated_dir, random_image)

print(f"Annotated image saved at: {annotated_image_path}")

#final image display - result
display(Image(filename=annotated_image_path))


import os, glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#Path of the image
image_path = r"E:\DS project\YOLO_SIMPLE_SHAPES\download.png"

if not os.path.exists(image_path):
    print(f" File not found: {image_path}")
else:
    print(f" Running inference on: {image_path}")

    results = model.predict(
        source=image_path,
        conf=0.5,
        save=True,
        imgsz=640,
        device='cpu'
    )

    annotated_dir = results[0].save_dir
    annotated_images = glob.glob(os.path.join(annotated_dir, "*.jpg")) + glob.glob(os.path.join(annotated_dir, "*.png"))

    if annotated_images:
        latest_image = max(annotated_images, key=os.path.getctime)
        print(f"\n Annotated image saved at: {latest_image}\n")

        #  Displaying the image using Matplotlib with large size
        img = mpimg.imread(latest_image)
        plt.figure(figsize=(8, 4))  # (width, height) in inches
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        print(" No annotated image found in:", annotated_dir)



