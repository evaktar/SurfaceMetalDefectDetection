from ultralytics import YOLO

''' Export best YOLO model for Android app '''

# Load best trained model
model = YOLO(r"runs\detect\yolo11n_aug_oversample\weights\best.pt")

# Export to TFLite with NMS for android
model.export(
    format="tflite",
    imgsz=640,
    int8=False,
    nms=True,
    data="config.yaml"
)
# yolo detect export model=runs\detect\train_yolo11n\weights\best.pt imgsz=320 batch=1 format=tflite