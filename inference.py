
# Import Libraries
import os
import glob
import random
from PIL import Image
# Predict
import torch
import torchvision.transforms as T
from ultralytics import YOLO
from load_models import load_ssd, load_fasterrcnn, load_retinanet
# Plot
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pipeline import get_reverse_mapping

''' Inference Visualisations and Plots '''

#__________ CONFIGURATIONS __________

# Run on correct directory
cur_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cur_dir)
# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
# Data info
image_size = 640
num_classes = 7
base_path = "NEU-DET"
test_images_path = os.path.join(base_path, "test", "images")
test_labels_path = os.path.join(base_path, "test", "labels")
os.makedirs("metrics", exist_ok=True) # Ensure exists
transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
# Class info
id_to_class = get_reverse_mapping()  # {0: "crazing", ...}
class_names = [id_to_class[i] for i in sorted(id_to_class.keys())]
# Detection box colour scheme
class_colors = {
    0: (0, 70, 220),
    1: (0, 140, 170),
    2: (90, 90, 90),
    3: (0, 150, 120),
    4: (25, 45, 130),
    5: (190, 50, 160),
}

#__________ HELPERS __________

# Get corresponding label to an image
def get_label_path(img_path):
    # Get subfolder from full image path
    img_dir, img_file = os.path.split(img_path)  
    # Get class name (subfolder name)
    cls_name = os.path.basename(img_dir)    
    # Image name from full path
    stem = os.path.splitext(img_file)[0]   
    return os.path.join(test_labels_path, cls_name, stem + ".txt")

# Get defect classes
def parse_label_file(label_path):
    classes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.append(int(parts[0]))  # Class ID e.g. 1 0.567 0...
    return classes

# Build list of tuples of image and label paths
def get_test_data():
    data = []
    # Recursive folder search
    pattern = os.path.join(test_images_path, "**", "*.jpg")
    # Build images and labels list
    for img_path in sorted(glob.glob(pattern, recursive=True)):
        label_path = get_label_path(img_path)
        if os.path.exists(label_path):
            data.append((img_path, label_path))
    return data

# Retrieve PIL image dimensions
def load_image_data(img_path):
    # Open PIL (from folder)
    img_pil = Image.open(img_path).convert("RGB")
    img_rgb = np.array(img_pil) # to numpy array for processing
    # Get sizes
    img_w, img_h = img_pil.size
    return img_pil, img_rgb, img_w, img_h

def label_format(img, text, x1, y1, color,
                 font=cv2.FONT_HERSHEY_SIMPLEX,font_scale=0.4, thickness=1, pad=2):
    # Size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # Add text to the top of the detection box
    text_bottom = y1 - 4
    box_top = y1 - text_height - baseline - 2 * pad - 4
    box_bottom = y1
    if box_top < 0:
        box_top = y1
        box_bottom = y1 + text_height + baseline + 2 * pad
        text_bottom = box_bottom - baseline - pad
    # Label colour and text
    cv2.rectangle(
        img,(x1, box_top), (x1 + text_width + 2 * pad, box_bottom), color, thickness=-1
    )
    cv2.putText(
        img, text, (x1 + pad, text_bottom - pad), font,
        font_scale, (255, 255, 255), thickness, cv2.LINE_AA
    )

def ground_truth_boxes(img_rgb, label_path, class_map):
    # Image
    plotted = img_rgb.copy()
    height, width = plotted.shape[:2]
    if not os.path.exists(label_path):
        return plotted # No truth box
    # Ground truths
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            # Class and bounding box coordinates
            cls_id, x_c, y_c, bw, bh = map(float, parts)
            cls_id = int(cls_id)
            # Bounding box coordinates * image size
            x_c *= width
            y_c *= height
            bw *= width
            bh *= height
            # Plot points
            x1 = int(x_c - bw / 2)
            y1 = int(y_c - bh / 2)
            x2 = int(x_c + bw / 2)
            y2 = int(y_c + bh / 2)

            # Colour to class
            cls_name = class_map.get(cls_id, str(cls_id))
            color = class_colors.get(cls_id, (60, 60, 60))
            # Plot ground truth
            cv2.rectangle(plotted, (x1, y1), (x2, y2), color, 2)
            label_format(plotted, cls_name, x1, y1, color)

    return plotted

# Draw box on detection
def draw_detections(img, boxes, labels, scores):
    # For each box
    for box, cls_id, score in zip(boxes.astype(int), labels, scores):
        x1, y1, x2, y2 = box
        cls_name = id_to_class.get(int(cls_id))
        color = class_colors.get(int(cls_id))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label_format(img, f"{cls_name} {score:.2f}", x1, y1, color)
    # Return image
    return img

# Draw prediction boxes 
def prediction_boxes(model_type, model, img_path, img_rgb, conf=0.25):
    # Shallow copy RGB array 
    plotted = img_rgb.copy()
    boxes, labels, scores = get_predictions(model_type, model, img_path, conf=conf)
    # Add predicted box and label  
    plotted = draw_detections(plotted, boxes, labels, scores)
    return plotted

# Save plot to metrics folder
def save_figure(fig, save_path):
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}.")

#__________ LOAD MODELS __________

def load_models(yolo_mappings, torch_mappings, conf=0.05, iou=0.5):
    models = {} # All models

    # YOLO models
    for name, path in yolo_mappings.items():
        # Load saved YOLO model
        models[name] = ("yolo", YOLO(path))

    # For each torch model
    weights_path = "model_weights"
    for filename in sorted(os.listdir(weights_path)):
        if not filename.endswith(".pt"):
            continue
        # Get model name
        stem = os.path.splitext(filename)[0].lower()
        name, loader = torch_mappings[stem]
        model = loader()
        state_dict = torch.load(os.path.join(weights_path, filename), map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        # NMS threshold (IoU) and Confidence score
        if hasattr(model, "roi_heads"):   # Faster R-CNN
            model.roi_heads.score_thresh = conf
            model.roi_heads.nms_thresh = iou
        else: # SSD and RetinaNet
            model.score_thresh = conf
            model.nms_thresh = iou
        model.to(device).eval() # Evaluate 
        models[name] = ("torchvision", model)

    return models

#__________ PREDICTIONS __________

# YOLO model predicted classes
def yolo_predict(model, img_path, conf=0.25):
    # Predict and retrieve boxes, labels, scores
    result = model.predict(str(img_path), conf=conf, verbose=False, iou=0.5)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return (np.zeros((0, 4), dtype=float), 
                np.zeros((0,), dtype=int), 
                np.zeros((0,), dtype=float))
    # Convert tensors to numpy array
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    labels = result.boxes.cls.detach().cpu().numpy().astype(int)
    scores = result.boxes.conf.detach().cpu().numpy()

    return boxes, labels, scores

# SSD, Faster-RCNN, RetinaNet inference
def torchvision_predict(model, img_path, conf=0.25):
    # Tensors to GPU
    img, _, w, h = load_image_data(img_path)
    img_tensor = transform(img).unsqueeze(0).to(device)
    # Prediction
    with torch.no_grad():
        preds = model(img_tensor)[0]
    # Convert tensors to numpy array 
    boxes = preds["boxes"].detach().cpu().numpy()
    labels = preds["labels"].detach().cpu().numpy().astype(int) - 1
    scores = preds["scores"].detach().cpu().numpy()
    # Threshold check
    keep = scores >= conf
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    if len(boxes) > 0: # Rescale
        scale_x = w / image_size
        scale_y = h / image_size
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

    return boxes, labels, scores

# Predictions 
def get_predictions(model_type, model, img_path, conf=0.25):
    # Get detections
    if model_type == "yolo":
        return yolo_predict(model, img_path, conf)
    elif model_type == "torchvision":
        return torchvision_predict(model, img_path, conf)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

#__________ CONFUSION MATRIX __________

# Compute confusion matrix per model
def compute_confusion_matrix(model_type, model, test_data):
    all_true, all_pred = [], []
    # For each data sample
    for img_path, label_path in test_data:
        # Ground truth classes 
        true_classes = parse_label_file(label_path)
        # Predicted classes
        _, labels, _ = get_predictions(model_type, model, img_path)
        pred_classes = labels.tolist()
        # Get list with missed/extra pred as max
        max_len = max(len(true_classes), len(pred_classes))
        # If < max then add padding (bg)
        all_true.extend(true_classes + [num_classes-1] * (max_len - len(true_classes)))
        all_pred.extend(pred_classes + [num_classes-1] * (max_len - len(pred_classes)))
    return confusion_matrix(all_true, all_pred, labels=list(range(num_classes)))

# Confusion matrix plotting
def plot_matrices(models, test_data, class_names):
    # Detection labels
    labels = class_names + ["background"]

    # Subplots row
    n = len(models)
    n_cols = 3
    n_rows = math.ceil(n / n_cols) # models/ columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    # Plot matrix per model
    for ax, (model_name, (model_type, model)) in zip(axes, models.items()):
        # Compute matrix
        cm = compute_confusion_matrix(model_type, model, test_data)
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels,
                    cbar=False, ax=ax)
        ax.set_title(model_name, fontsize=8)
        ax.tick_params(axis='x', rotation=45)
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    # Shared axis labels
    fig.supxlabel("Predicted", fontsize=10)
    fig.supylabel("True", fontsize=10)
    # Remove unused subplots
    for ax in axes[len(models):]:
        ax.set_axis_off()

    # Save plot
    save_figure(fig, "metrics/confusion_matrices.png")

#__________ PR Curve __________  

def parse_boxes(label_path, img_w, img_h):
    boxes, labels = [], []
    if not os.path.exists(label_path): # No detections
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=int)
    # For each label
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            # ID and YOLO-style coordinates
            cls_id, x_c, y_c, bw, bh = map(float, parts)
            cls_id = int(cls_id)
            # Absolute center pixel coordinates
            x_c *= img_w
            y_c *= img_h
            bw *= img_w
            bh *= img_h
            # Corner pixel coordinates of bboxes
            x1 = x_c - bw / 2
            y1 = y_c - bh / 2
            x2 = x_c + bw / 2
            y2 = y_c + bh / 2
            boxes.append([x1, y1, x2, y2])
            labels.append(cls_id)
    # Return as numpy arrays for processing
    return np.array(boxes, dtype=float), np.array(labels, dtype=int)

# Get IoU of overlapping boxes
def box_iou_np(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=float)
    # Find intersection rectangle
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    # Intersection
    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    # Intersection area
    inter = inter_w * inter_h

    # Union
    # Area of box being compared
    box_area = max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))
    # Area of boxes compared to
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    # Union area 
    union = box_area + boxes_area - inter + 1e-9

    return inter / union # IoU

# Calc PR
def compute_pr_curve(model_type, model, test_data, iou_thresh=0.5, conf=0.01):
    detections = []
    gt_count = 0

    # For each sample
    for img_path, label_path in test_data:
        _, _, img_w, img_h = load_image_data(img_path)
        # Ground truth and prediction detections 
        gt_boxes, gt_labels = parse_boxes(label_path, img_w, img_h)
        pred_boxes, pred_labels, pred_scores = get_predictions(model_type, model, img_path, conf=conf)
        gt_count += len(gt_labels)
        gt_used = np.zeros(len(gt_labels), dtype=bool)
        # Sort by scores attained  (descending)
        order = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]
        pred_scores = pred_scores[order]
        # Match predictions and ground truth
        for box, cls_id, score in zip(pred_boxes, pred_labels, pred_scores):
            same_class = np.where(gt_labels == cls_id)[0]
            if len(same_class) == 0:
                detections.append((score, 0))
                continue
            candidate_boxes = gt_boxes[same_class]
            # Get best predicted to ground truth
            ious = box_iou_np(box, candidate_boxes)
            best_local = np.argmax(ious)
            best_iou = ious[best_local]
            best_gt_idx = same_class[best_local]
            # Keep only if IoU threshold met
            if best_iou >= iou_thresh and not gt_used[best_gt_idx]:
                gt_used[best_gt_idx] = True
                detections.append((score, 1))
            else:
                detections.append((score, 0))

    if len(detections) == 0: # No detections
        return np.array([0.0]), np.array([0.0]), 0.0
    
    # Sort by confidence score 
    detections.sort(key=lambda x: x[0], reverse=True)
    # Get measures from matches
    tp = np.array([d[1] for d in detections], dtype=float)
    fp = 1.0 - tp
    # Sum TP/FP 
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    # Precision and recall accumulated
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    recall = tp_cum / (gt_count + 1e-9)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    # Smoothing 
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    # Compute AP (approx area under PR curve)
    ap = np.trapezoid(mpre, mrec)
    return recall, precision, ap

# Plot PR
def plot_pr_curves(models, test_data, iou_thresh=0.5, conf=0.001):
    fig, ax = plt.subplots(figsize=(8, 6))
    # For each model
    for model_name, (model_type, model) in models.items():
        # Compute PR Curve 
        recall, precision, ap = compute_pr_curve(
            model_type=model_type, model=model, test_data=test_data,
            iou_thresh=iou_thresh, conf=conf)
        # Plot PR Curve
        ax.plot(recall, precision, linewidth=2, label=f"{model_name} (AP@0.5={ap:.3f})")
    # PR and IoU Thresh
    ax.set_title(f"Precision-Recall Curve (IoU={iou_thresh:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="lower left")
    # Save figure 
    save_figure(fig, "metrics/pr_curves.png")

#__________ DETECIONS PLOT __________

def per_class_samples():
    # Per-class sample images
    sample_data = []
    # Randomise images (reproducible)
    random.seed(42)

    # Get class folders from test images directory
    class_dirs = sorted([
        d for d in os.listdir(test_images_path)
        if os.path.isdir(os.path.join(test_images_path, d))
    ])
    # For each class 
    for cls_name in class_dirs:
        class_dir = os.path.join(test_images_path, cls_name)
        # Get all images in class folder
        img_paths = sorted([
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith(".jpg")
        ])
        # Randomise images 
        random.shuffle(img_paths)
        data = None

        # For each JPG image (one per class)
        for img_path in img_paths:
            img_bgr = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # Get corresponding label
            label_path = get_label_path(img_path)
            # Add data info
            data = {
                "img_path": str(img_path),
                "img_rgb": img_rgb,
                "label_path": label_path,
                "title": cls_name,
            }
            break

        if data is not None:
            sample_data.append(data)
    return sample_data


def plot_detection_results(conf=0.25):
    # Sample images per-class
    sample_data = per_class_samples()
    n_samples = len(sample_data)
    n_cols = n_samples
    n_rows = 2 + len(models)

    # Create plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    # Original Row 1
    for col, sample in enumerate(sample_data):
        axes[0, col].imshow(sample["img_rgb"])
        axes[0, col].set_title(sample["title"], fontsize=13)  # Class name

    # Ground truth Row 2
    for col, sample in enumerate(sample_data):
        gt_img = ground_truth_boxes(sample["img_rgb"], sample["label_path"], id_to_class)
        axes[1, col].imshow(gt_img)

    # Models n Rows
    for row_idx, (display_name, (model_type, model)) in enumerate(models.items(), start=2):
        for col, sample in enumerate(sample_data):
            plotted = prediction_boxes(
                model_type, model, sample["img_path"], sample["img_rgb"], conf=conf
            )
            axes[row_idx, col].imshow(plotted)

    # Row labels
    axes[0, -1].set_ylabel("Original", fontsize=14, rotation=270, labelpad=20)
    axes[0, -1].yaxis.set_label_position("right")
    axes[1, -1].set_ylabel("Ground Truth", fontsize=14, rotation=270, labelpad=20)
    axes[1, -1].yaxis.set_label_position("right")

    # Model names
    for row_idx, display_name in enumerate(models.keys(), start=2):
        axes[row_idx, -1].set_ylabel(display_name, fontsize=14, rotation=270, labelpad=20)
        axes[row_idx, -1].yaxis.set_label_position("right")

    # Remove axes (detection image not a graph)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.suptitle("Detection Results", fontsize=16, y=0.99)  # y-offset

    # Save figure
    save_figure(fig, "metrics/detection_results.png")
    print(f"\nDiagrams in metrics folder.")


''' These are plot manually as they are adhoc visualisations '''

# mAP50 against FPS
model_stats = {
    "YOLO11n": {"map50": 0.7615, "fps": 103.26},
    "Faster R-CNN": {"map50": 0.7013, "fps": 15.61},
    "SSD": {"map50": 0.5492, "fps": 81.86},
    "RetinaNet": {"map50": 0.5364, "fps": 19.17},
    "YOLO11n-AO": {"map50": 0.7740, "fps": 78.61},
    "YOLO11n-MobileNetv3": {"map50": 0.6742, "fps": 80.79},
    "YOLO11n-CBAM": {"map50": 0.7073, "fps": 57.49},
}
# Per-Class Metrics 
per_class_metrics = {
    "crazing":          {"ap": 0.1842, "p": 0.6301, "r": 0.3235},
    "inclusion":        {"ap": 0.4756, "p": 0.9022, "r": 0.7258},
    "patches":          {"ap": 0.6342, "p": 0.9124, "r": 0.8419},
    "pitted_surface":   {"ap": 0.3966, "p": 0.9354, "r": 0.6293},
    "rolled-in_scale":  {"ap": 0.2183, "p": 0.6977, "r": 0.4493},
    "scratches":        {"ap": 0.4424, "p": 0.7091, "r": 0.8438},
}

#__________ FPS against mAP50 Line Graph __________

def plot_fps_map50(model_stats):
    # Plot line graph 
    fig, ax = plt.subplots(figsize=(8, 6))
    names = list(model_stats.keys())
    colors = plt.cm.tab10(np.arange(len(names)))

    # For each model in mapping
    for model_name, color in zip(names, colors):
        # FPS against mAP50
        x = model_stats[model_name]["fps"]
        y = model_stats[model_name]["map50"]
        # Scatter points
        ax.scatter(
            x, y, s=130, color=color, 
            edgecolors="black", linewidths=0.6
        )
        # Model Name
        ax.annotate(
            model_name, (x, y),
            textcoords="offset points",
            xytext=(0, 8), ha="center", 
            va="bottom", fontsize=7
        )
    # Labels
    ax.set_title("FPS vs mAP@0.5")
    ax.set_xlabel("FPS")
    ax.set_ylabel("mAP@0.5")
    ax.grid(True, linestyle="--", alpha=0.4)
    # Save figure
    save_figure(fig, "metrics/fps_vs_map50.png")

#__________ PER_CLASS P+R Stacked Bar Chart __________ 

def plot_per_class_pr_stacked(per_class_metrics):
    # Metrics from mappings
    classes = list(per_class_metrics.keys())
    precision = [per_class_metrics[c]["p"] for c in classes]
    recall    = [per_class_metrics[c]["r"] for c in classes]

    # Barcharts (stacked precision and recall)
    y = np.arange(len(classes))
    height = 0.6
    fig, ax = plt.subplots(figsize=(4, 8))
    ax.barh(
        y, precision, height,
        label="Precision", color="#4C72B0"
    )
    ax.barh(
        y, recall,height, left=precision,
        label="Recall", color="#9ECAE1"
    )
    # Labels
    ax.set_title("Per-class Precision and Recall YOLO11n-AO")
    ax.set_yticks(y)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Score")
    ax.set_xlim(0, 2.0)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend()
    # Scores on bar charts
    for i, (p, r) in enumerate(zip(precision, recall)):
        ax.text(
            p / 2, y[i],  f"{p:.2f}", 
            ha="center", va="center",
            fontsize=10, color="white"
        )
        ax.text(
            p + r / 2, y[i], f"{r:.2f}",
            ha="center", va="center",
            fontsize=10, color="black"
        )
    ax.invert_yaxis()  # top-to-bottom class order
    # Save figure
    save_figure(fig, "metrics/per_class_pr_stacked.png")

#__________ MAIN __________  

# Data from test folder
test_data = get_test_data()
print(f"Retrieving {len(test_data)} test data.")

# Models to plot 
base_yolo = "runs/detect/"
yolo_mappings = {
    "YOLO11n": base_yolo + "yolo11n/weights/best.pt",
    "YOLO11n-AO": base_yolo + "yolo11n_aug_oversample/weights/best.pt",
    "YOLO11n CBAM": base_yolo + "yolo11n_cbam/weights/best.pt",
    "YOLO11n MobileNetv3": base_yolo + "yolo11n_mobilenetv3/weights/best.pt",
}
torch_mappings = {
    "ssd": ("SSD", lambda: load_ssd(num_classes, image_size)),
    "fasterrcnn": ("Faster R-CNN", lambda: load_fasterrcnn(num_classes)),
    "retinanet": ("RetinaNet", lambda: load_retinanet(num_classes)),
}

# Plots
models = load_models(yolo_mappings, torch_mappings)
plot_matrices(models, test_data, class_names) # Confusion Matrix
plot_pr_curves(models, test_data) # PR Curve
# Detections
plot_detection_results() # Inference image results (boxes, labels, and conf)
# Manual Plots
plot_fps_map50(model_stats)
plot_per_class_pr_stacked(per_class_metrics)