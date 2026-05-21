# Import Libraries
import os # File paths
import torch
from torch import nn
from ultralytics import YOLO # Load model
import pandas as pd # Read results as df
import time # FPS
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from fvcore.nn import FlopCountAnalysis # Calc FLOPs

''' Retrieve and Compute Metrics '''

# Helper for FPS and GFLOPs estimates 
class InferenceWrapper(nn.Module):
    def __init__(self, model, model_name):
        super().__init__()
        self.model = model
        self.model_name = model_name # e.g., "yolo"
    def forward(self, x):
        if self.model_name == "yolo":
            # YOLO (b, 3, h, w)
            return self.model(x)
        else:
            # Torchvision - list[Tensor]
            return self.model([x[0]])

def calc_fps(model, device, dummy_input):
    # FPS
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    # Timing
    runs = 100
    # GPU Sync
    torch.cuda.synchronize() if device.type == "cuda" else None
    # Start time
    start = time.perf_counter()
    # Dummy model inference
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
    # GPU Sync
    torch.cuda.synchronize() if device.type == "cuda" else None
    # End time
    end = time.perf_counter()
    # Latency
    lat = (end - start) / runs
    # FPS
    fps = 1 / lat

    return fps, lat

# YOLO train/validation metrics
def yolo_train_metrics(metrics_path, device, size, train_path):
    # Load the trained model
    model_path = f"{train_path}/weights/best.pt"
    load_model = YOLO(model_path)
    # Evaluation mode
    model = load_model.model.to(device)
    model.eval()

    # Get model info
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    params = sum(p.numel() for p in model.parameters())
    # Metrics of last epoch
    df = pd.read_csv(f"{train_path}/results.csv")
    last = df.iloc[-1]
    # Get precision and recall metrics (+used to calc F1)
    precisionGet = last.get("metrics/precision(B)")
    recallGet = last.get("metrics/recall(B)")
    # FLOPs/FPS wrapper 
    wrapped_model = InferenceWrapper(model, "yolo").to(device).eval()
    dummy_input = torch.randn(1, 3, size, size, device=device)
    # FLOPs with fvcore
    flops = FlopCountAnalysis(wrapped_model, dummy_input) \
        .unsupported_ops_warnings(False) \
        .uncalled_modules_warnings(False) \
        .tracer_warnings("none")
    gflops = flops.total() / 1e9
    # Calc FPS
    fps, lat = calc_fps(wrapped_model, device, dummy_input)
    # Summary of metrics
    summary = {
        "Size (MB)": f"{size_mb}",
        "Params": params,
        "mAP@0.5": last.get("metrics/mAP50(B)"),
        "mAP@0.5:0.95": last.get("metrics/mAP50-95(B)"),
        "F1": 2 * (precisionGet * recallGet) / (precisionGet + recallGet) if (precisionGet + recallGet) > 0 else 0,
        "Precision": precisionGet,
        "Recall": recallGet,
        "train_loss": last.get("train/box_loss") + last.get("train/cls_loss"),
        "val_loss": last.get("val/box_loss") + last.get("val/cls_loss"), 
        "GFLOPs": gflops,
        "FPS": fps,
        "Latency": lat,
    }
    # Write to file and print to console (monitoring)
    print("\nYOLO Training and Validation")
    with open(metrics_path, "a+", encoding="utf-8") as f:
        f.write("\nYOLO Training and Validation\n")
        for k, v in summary.items():
            if isinstance(v, float):
                v = round(v, 4)
            line = f"{k}: {v}\n"
            print(line.strip()) # Print 
            f.write(line)

# Metrics for models except YOLO
# Equations
def compute_precision_recall_f1(tp, fp, fn, eps=1e-9):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1

def match_predictions(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh):
    # If no ground truths
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0 # All are FP 

    # Get IoU of predictions and ground truths
    ious = box_iou(pred_boxes, gt_boxes)
    matched_gt = set() # 1 prediction to 1 ground truth

    tp = 0
    fp = 0

    # Retrieve highest IoU ground truth per prediction (best match)
    for i in range(len(pred_boxes)):
        max_iou, idx = ious[i].max(0)
        idx = idx.item()
        #  True positive if conditions met
        if (
            max_iou >= iou_thresh # Matching or higher IoU
            and idx not in matched_gt # Not already matched
            and pred_labels[i] == gt_labels[idx] # Classes match
        ):
            tp += 1
            matched_gt.add(idx)
        else: # False positive 
            fp += 1

    # Ground truth boxes that have not been predicted
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn

# Calc
def get_metrics(model, dataloader, iou_thresh=0.5, score_thresh=0.5):
    device = next(model.parameters()).device
    map_metric = MeanAveragePrecision(iou_type="bbox").to(device)
    model.eval() # Evaluation mode
    # Measures
    tp = 0
    fp = 0
    fn = 0

    # For each defect
    with torch.no_grad():
        for images, targets in dataloader:
            # img to GPU (device)
            if isinstance(images, list):
                images = [img.to(device) for img in images]
            else:
                images = images.to(device)
            # Targets to GPU
            targets_list = []
            if isinstance(targets, dict):
                for boxes, labels in zip(targets["bbox"], targets["cls"]):
                    targets_list.append({
                        "boxes": boxes.to(device),
                        "labels": labels.to(device),
                    })
            else: # Tensor processing
                targets_list = [
                    {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                    for t in targets
                ]

            outputs = model(images)# Detections
            preds_batch, grts_batch = [], []
            # For each prediction
            for pred, target in zip(outputs, targets_list):
                if isinstance(pred, dict) and "scores" in pred:
                    # Bounding boxes over threshold
                    keep = pred["scores"] > score_thresh
                    pred_boxes = pred["boxes"][keep].cpu()
                    pred_labels = pred["labels"][keep].cpu()
                    pred_scores = pred["scores"][keep].cpu()
                else: # No prediction (empty tensor)
                    pred = pred.detach()
                    if pred.numel() == 0:
                        pred_boxes = torch.zeros((0, 4))
                        pred_scores = torch.zeros((0,))
                        pred_labels = torch.zeros((0,), dtype=torch.long)
                    else: # Tensor preocessing
                        pred_boxes = pred[:, :4].cpu()
                        pred_scores = pred[:, 4].cpu()
                        pred_labels = pred[:, 5].cpu().long()
                # Ground truth
                gt_boxes = target["boxes"].cpu()
                gt_labels = target["labels"].cpu()

                # TP, FP, FN from ground truth to prediction matches
                d_tp, d_fp, d_fn = match_predictions(
                    pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh
                )
                tp += d_tp
                fp += d_fp
                fn += d_fn

                # Predictions batch
                preds_batch.append({
                    "boxes": pred_boxes,
                    "scores": pred_scores,
                    "labels": pred_labels,
                }) # Ground truth
                grts_batch.append({
                    "boxes": gt_boxes,
                    "labels": gt_labels,
                })

            # Update mAP with batches
            map_metric.update(preds_batch, grts_batch)

    # Compute metrics
    precision, recall, f1 = compute_precision_recall_f1(tp, fp, fn)
    # Compute mAP
    result = map_metric.compute()
    map_50_95 = float(result["map"])
    map_50 = float(result["map_50"])

    return precision, recall, f1, map_50, map_50_95

# All metrics for the models
def final_metrics(model, device, size, best_model, metrics_path, val_loader, test_loader, yamlPath, batch, class_mappings=None, model_name=None):
    if model_name != "yolo":
        # Load weights
        state_dict = torch.load(best_model, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Model Size (MB)
    size_mb = os.path.getsize(best_model) / (1024 * 1024)
    # Parameter Count
    params = sum(p.numel() for p in model.parameters())

    # Build inference models for GFLOPs and FPS
    if model_name == "yolo":
        infer_model = model.model.to(device).eval()
    else:
        infer_model = model

    # Wrapper for FLOPs and FPS
    wrapped_model = InferenceWrapper(infer_model, model_name).to(device).eval()
    
    # GFLOPS estimate
    dummy_input = torch.randn(1, 3, size, size, device=device)
    flops = FlopCountAnalysis(wrapped_model, dummy_input) \
            .unsupported_ops_warnings(False) \
            .uncalled_modules_warnings(False) \
            .tracer_warnings("none")
    gflops = flops.total() / 1e9
    # Calc FPS
    fps, lat = calc_fps(wrapped_model, device, dummy_input)

    if model_name != "yolo":
        loader_mappings = {
            "Validation": val_loader,
            "Test": test_loader
        }

        # Precision, Recall, F1
        for name, loader in loader_mappings.items():
                precision, recall, f1, map5, map5_95 = get_metrics(infer_model, loader)
                # Write metrics to file
                with open(metrics_path, "a+") as f:
                    f.write(f"\n{model_name} {name}\n")
                    f.write(f"Size (MB): {size_mb:.4f}\n")
                    f.write(f"Params: {params}\n")
                    f.write(f"mAP@0.5: {map5:.4f}\n")
                    f.write(f"mAP@0.5:0.95: {map5_95:.4f}\n")
                    f.write(f"F1: {f1:.4f}\n")
                    f.write(f"Precision: {precision:.4f}\n")
                    f.write(f"Recall: {recall:.4f}\n")
                    f.write(f"GFLOPs: {gflops:.4f}\n")
                    f.write(f"FPS: {fps:.4f}\n")
                    f.write(f"Latency: {lat:.4f}\n\n")
    else: # Test run
        test_results = model.val(
            data=yamlPath,
            split="test",
            imgsz=size,
            batch=batch,
            workers=0,
            device=device,
            cache=False,
            plots=True,
            project="val_fine_aug"
        )
        # Test metrics
        precision = test_results.box.p.mean()
        recall = test_results.box.r.mean()
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        class_id_mappings = class_mappings

        # Write to metrics file
        with open(metrics_path, "a+") as f:
            f.write(f"\n{model_name} Test\n")
            f.write(f"Size (MB): {size_mb:.4f}\n")
            f.write(f"Params: {params}\n")
            f.write(f"mAP@0.5: {test_results.box.map50:.4f}\n")
            f.write(f"mAP@0.5:0.95: {test_results.box.map:.4f}\n")
            f.write(f"F1: {f1:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"GFLOPs: {gflops:.4f}\n")
            f.write(f"FPS: {fps:.4f}\n")
            f.write(f"Latency: {lat:.4f}\n\n")
            
            f.write("PER-CLASS METRICS\n")
            # For each class
            for cid, cname in class_id_mappings.items():
                # Retrieve the computed metrics
                ap = test_results.box.maps[cid]
                p = test_results.box.p[cid]
                r = test_results.box.r[cid]
                f.write(f"{cname}: AP={ap:.4f}, P={p:.4f}, R={r:.4f}\n")
