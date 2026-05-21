# Import libraries
import subprocess
import os

''' Run pipeline for multiple models '''

# Run on correct directory
cur_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cur_dir)

yolo_models = [ 
    "yolov11n",
    "yolo11n_mobilenetv3","yolo11n_cbam",
    "yolo11n_cbam_default",
    "yolov5n",  "yolov8n", "yolo12n",
    "yolov5s", "yolov8s", "yolo11s", "yolo12s",
    "yolov5m", "yolov8m"
]
torch_models = [
    "ssd", 
    "fasterrcnn", 
    "retinanet"
]

# Run YOLO
for model in yolo_models:
    cmd = [
        "python",
        "pipeline.py",
        "--augmentation",
        "--run_yolo",
        "--yolo_model", model,
        "--epochs", "150",
        "--yolo_batch", "12",
        "--patience", "30",
        "--workers", "4"
    ]
    subprocess.run(cmd, check=True, cwd=cur_dir)

# Run Torch 
for model in torch_models:
    cmd = [
        "python",
        "pipeline.py",
        "--run_torch",
        "--torch_models", model,
        "--epochs", "1",
        "--torch_batch", "2",
        "--patience", "30",
        "--workers", "4"
    ]
    subprocess.run(cmd, check=True, cwd=cur_dir)