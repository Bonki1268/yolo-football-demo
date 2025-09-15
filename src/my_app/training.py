from ultralytics import YOLO
# import torch

# print(torch.cuda.is_available())
# print(torch.backends.mps.is_available())

# model = YOLO("yolov8n")
model = YOLO(
    "/Users/sam/Documents/Project/basketball/yolo_demo/my-app/models/best.pt")

results = model(
    source="/Users/sam/Documents/Project/basketball/yolo_demo/my-app/videos/football.mp4",
    save=True, conf=0.25, show=True, device="mps")

print(results[0])
print("==============")
