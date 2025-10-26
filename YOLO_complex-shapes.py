import torch
from ultralytics import YOLO

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