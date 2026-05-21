# Import Libraries
import os
import argparse # Automated training
from pathlib import Path
# Training
import torch
from ultralytics import YOLO
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.amp import autocast, GradScaler
# Helper functions
# Dataset
from dataprep import data_resplit, xmlToTxt, oversample, get_custom_aug
from dataset import create_dataloader
# Models
from load_models import get_optimizer, load_ssd, load_fasterrcnn, load_retinanet
from metrics import yolo_train_metrics, final_metrics # Evaluation

''' Main Pipeline '''

#__________ OVERALL __________

# Ensure running on correct directory
cur_dir = Path.cwd()
os.chdir(cur_dir)
# All paths
dataset_path = cur_dir / "NEU-DET"
yaml_path = cur_dir / "config" / "config.yaml"
metrics_path = cur_dir / "metrics"
best_model_path = cur_dir / "model_weights"
# Ensure exists to save outcomes
metrics_path.mkdir(exist_ok=True)
best_model_path.mkdir(exist_ok=True)

# Get split paths
def get_dataset_splits(dataset_path):
    train_path = dataset_path / "train"
    val_path = dataset_path / "validation"
    test_path = dataset_path / "test"
    return train_path, val_path, test_path

# Save metrics to appropriate path
def get_metrics_path(model_name):
    return metrics_path / f"{model_name}.txt"

# Classes and indexes
def get_mapping():
    return {
        "crazing": 0,
        "inclusion": 1,
        "patches": 2,
        "pitted_surface": 3,
        "rolled-in_scale": 4,
        "scratches": 5,
    }
# Indexes to classes
def get_reverse_mapping():
    return {v: k for k, v in get_mapping().items()}

#__________ DATA PREPARATIONS/ PREPROCESSING ___________

# Dataset prep requirements
def prepare_dataset(dataset_path=dataset_path, resplit=True, convert_xml=True):
    # Required params
    train_path, val_path, test_path = get_dataset_splits(dataset_path)
    datasets = [train_path, val_path, test_path]
    class_mapping = get_mapping()
    class_folders = list(class_mapping.keys())
    folders = ["images", "labels", "annotations"]
    image_split_paths = [
        train_path / "images",
        val_path / "images",
        test_path / "images",
    ]

    # Does dataset need repslit
    if resplit:
        data_resplit(folders, image_split_paths, class_folders, val_path, test_path, 30)
    # Does dataset need TXT labels
    if convert_xml:
        xmlToTxt(class_mapping, datasets)

    # Oversample
    oversample(image_split_paths[0], train_path / "labels", train_path, target_class=(0, 4), duplicate=3)

#__________ YOLO __________

# Latest trained yolo
def latest_yolo_folder():
    # Saved results + model folder
    train_folder = cur_dir / "runs" / "detect"
    if not train_folder.exists():
        return None
    # YOLO subfolders
    subfolders = [p for p in train_folder.iterdir() if p.is_dir()]
    if not subfolders:
        return None

    return max(subfolders, key=lambda x: x.stat().st_ctime)

# Log YOLO optimiser configurations
def log_yolo_config(trainer, model_name):
    # Optimiser parameters
    opt = trainer.optimizer
    opt_param = opt.param_groups[0]
    # Learning rate and scheduler
    args = trainer.args
    metrics_path = get_metrics_path(model_name)

    with open(metrics_path, "a+", encoding="utf-8") as f:
        f.write(f"\n{model_name} OPTIMISER AND CONFIG\n")
        f.write(f"optimiser: {type(opt).__name__}\n")
        f.write(f"lr: {opt_param['lr']}\n")
        f.write(f"weight_decay: {opt_param['weight_decay']}\n")
        f.write(f"momentum: {opt_param.get('momentum', 'N/A')}\n")
        f.write(f"scheduler: {trainer.scheduler.__class__.__name__}\n")
        f.write(f"cos_lr: {getattr(args, 'cos_lr', None)}\n")
        f.write(f"lr0: {args.lr0}\n")
        f.write(f"lrf: {args.lrf}\n\n")

# Train YOLO models
def train_yolo(augmentation=True, model_name="yolo11n", epochs=150, size=640, batch=12, patience=30,
               optimizer="AdamW", lr0=0.001, lrf=0.01, weight_decay=0.0005,
               momentum=0.9, cos_lr=False, cache=False, workers=1, device=None):
    
    # Reset GPU memory
    torch.cuda.empty_cache()
    # Augmentations
    if augmentation:
        custom_aug=get_custom_aug()
    else:
        custom_aug=None

    # Model version
    if model_name == "yolo11n_mobilenetv3":
        model = YOLO("config/yolo11n_mobilenetv3.yaml") 
    elif model_name == "yolo11n_cbam":
        model = YOLO("config/yolo11n_cbam.yaml")
    elif model_name == "yolo11n_cbam_default":
        model = YOLO("config/yolo11n_cbam_default.yaml")
    else:
        model = YOLO(f"{model_name}.pt")

    # Return optimiser configurations (fine-tuning)
    # def log_config_callback(trainer):
    #     log_yolo_config(trainer, model_name)
    # model.add_callback("on_train_start", log_config_callback)

    # Additional parameters
    extra = {}
    if optimizer.lower() == "sgd":
        extra["momentum"] = momentum

    # Training
    run_name = f"{model_name}"
    model.train(
        # Configs
        data=str(yaml_path),
        epochs=epochs,
        imgsz=size,
        batch=batch,
        workers=workers,
        device=device,
        name=run_name,
        plots=True,
        seed=42,
        # Performance
        patience=patience,
        cache=cache,
        # Optimiser
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        weight_decay=weight_decay,
        cos_lr=cos_lr,
        # Warmup
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        **extra,
        augmentations=custom_aug
    )

    # Save train/val metrics
    train_path = latest_yolo_folder()
    #train_path = Path("runs/detect/yolo11n")
    metrics_path = get_metrics_path(model_name)
    yolo_train_metrics(metrics_path, device, size, train_path)
    # Test metrics
    best_model = train_path / "weights" / "best.pt"
    yolo_model = YOLO(str(best_model))
    class_mappings = get_reverse_mapping() # ID to Class
    final_metrics(yolo_model, device, size, str(best_model), str(metrics_path), 
                  None, None, str(yaml_path), 1, class_mappings, model_name="yolo",
    )

#__________ OTHER MODELS __________

# Train single epoch
def train_epoch(model, optimizer, data_loader, device, epoch, scaler):
    model.train()
    total_loss = 0
    total_class_loss = 0
    total_box_loss = 0
    zero = torch.tensor(0, device=device)
    # Convert tensors to scalar
    scalar = lambda x: (
        x if torch.is_tensor(x) and x.numel() == 1
        else torch.tensor(float(x), device=device)
    )

    # Warmup
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for _, (images, targets) in enumerate(data_loader):
        optimizer.zero_grad(set_to_none=True) # Reset
        # Dict of bbox and cls lists 
        targets_list = []

        # Move image and target tensors to GPU
        if isinstance(targets, list): # SSD/ Faster R-CNN format
            # Targets and images to GPU 
            images = [img.to(device) for img in images]
            targets_list = [
                {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                for t in targets
            ]
            with autocast("cuda", enabled=scaler is not None):
                # Get train losses
                output = model(images, targets_list)
                # Get output (predictions)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                # Get loss values
                if "loss_classifier" in output or "loss_box_reg" in output:
                    # Faster R-CNN style
                    valid_keys = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
                    filtered = {k: scalar(v) for k, v in output.items() if k in valid_keys}
                    loss = sum(filtered.values())
                    class_loss = filtered.get("loss_classifier", zero)
                    box_loss = filtered.get("loss_box_reg", zero)
                elif "classification" in output or "bbox_regression" in output:
                    # SSD style
                    valid_keys = ["classification", "bbox_regression"]
                    filtered = {k: scalar(v) for k, v in output.items() if k in valid_keys}
                    loss = sum(filtered.values())
                    class_loss = filtered.get("classification", zero)
                    box_loss = filtered.get("bbox_regression", zero)
                else: # Not expected keys, print ones there
                    raise ValueError(f"Unknown loss keys: {list(output.keys())}")
        else:
            print("Invalid format:", images, targets)
            
        # Increment total image losses
        total_loss += loss.item()
        total_class_loss += class_loss.item()
        total_box_loss += box_loss.item()

        # Update model weights
        prev_scale = scaler.get_scale()
        scaler.scale(loss).backward() # Backprop (gradient on loss)
        scaler.step(optimizer) # Grad descent on params
        scaler.update()
        new_scale = scaler.get_scale()
        
        # Update learning rate
        if lr_scheduler is not None and new_scale == prev_scale:
            lr_scheduler.step()

    # Write training metrics for monitoring
    n = len(data_loader)
    avg_loss = total_loss / n
    avg_class_loss = total_class_loss / n
    avg_box_loss = total_box_loss / n

    print(
        f"Epoch {epoch + 1} Train | "
        f"loss={avg_loss:.4f} | "
        f"class_loss={avg_class_loss:.4f} | "
        f"box_loss={avg_box_loss:.4f}"
    )
    return avg_loss

# Validation/ evaluate single epoch
@torch.no_grad()
def validation_epoch(model, val_loader, device, epoch, scaler=None):
    map = MeanAveragePrecision(iou_type="bbox").to(device)
    # Ignore warnings, handled by score_thresh
    map.warn_on_many_detections = False
    # Validation losses
    total_loss = 0
    total_class_loss = 0
    total_box_loss = 0
    score_thresh = 0.5
    # Convert tensors to scalar
    scalar = lambda x: (
        x if torch.is_tensor(x) and x.numel() == 1
        else torch.tensor(float(x), device=device)
    )
    zero = torch.tensor(0, device=device) # For 0

    for images, targets in val_loader:
        preds = []
        grts = []

        # SSD, Faster R-CNN
        if isinstance(images, list):
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [
                {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in t.items()}
                for t in targets
            ]
            # Train mode for validation loss
            model.train() 
            with autocast("cuda", enabled=scaler is not None):
                output = model(images, targets)
            model.eval() # Back to evaluation mode

            # Get output (predictions)
            if isinstance(output, (tuple, list)):
                    output = output[0]
            # Get loss values
            if "loss_classifier" in output or "loss_box_reg" in output:
                # Faster R-CNN style
                valid_keys = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
                filtered = {k: scalar(v) for k, v in output.items() if k in valid_keys}
                loss = sum(filtered.values())
                class_loss = filtered.get("loss_classifier", zero)
                box_loss = filtered.get("loss_box_reg", zero)
            elif "classification" in output or "bbox_regression" in output:
                # SSD style
                valid_keys = ["classification", "bbox_regression"]
                filtered = {k: scalar(v) for k, v in output.items() if k in valid_keys}
                loss = sum(filtered.values())
                class_loss = filtered.get("classification", zero)
                box_loss = filtered.get("bbox_regression", zero)
            else: # Not expected keys, print ones there
                raise ValueError(f"Unknown loss keys: {list(output.keys())}")
            
            # Evaluation detections (eval output)
            detections = model(images)
            for det, tgt in zip(detections, targets):
                if det["boxes"].numel() == 0:
                    boxes = torch.zeros((0, 4), device=device)
                    scores = torch.zeros((0,), device=device)
                    labels = torch.zeros((0,), dtype=torch.int64, device=device)
                else:
                    boxes = det["boxes"]
                    scores = det["scores"]
                    labels = det["labels"].to(torch.int64)
                # Discard low detections
                keep = scores > score_thresh
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                # Add to lists
                preds.append({
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                })
                grts.append({
                    "boxes": tgt["boxes"],
                    "labels": tgt["labels"].to(torch.int64),
                })
        else:
            print("Invalid model format:",  images, targets)

        # Aggregate losses of detections
        total_loss += loss.item()
        total_class_loss += class_loss.item()
        total_box_loss += box_loss.item()

        map.update(preds=preds, target=grts)

    # Get average loss
    n = max(len(val_loader), 1)
    avg_loss = total_loss / n
    avg_class_loss = total_class_loss / n
    avg_box_loss = total_box_loss / n

    print(
        f"Epoch {epoch + 1} Val   | "
        f"loss={avg_loss:.4f} | "
        f"class_loss={avg_class_loss:.4f} | "
        f"box_loss={avg_box_loss:.4f}"
    )

    map_results = map.compute()
    return avg_loss, map_results

#__________ Faster R-CNN, SSD and RetinaNet __________

# Training loops
def train_torch_detector(model_name="fasterrcnn", epochs=30,
    size=640, batch=2, patience=30, optimizer="SGD", 
    lr=0.01, weight_decay=0.0005, momentum=0.9, device=None, workers=1):

    # Metrics and best model saving
    num_classes = 7  # 6 classes + 1 for bg
    metrics_path = get_metrics_path(model_name)
    best_model = best_model_path / f"{model_name}.pt"

    # Load correct model
    if model_name == "ssd":
        model = load_ssd(num_classes, size)
    elif model_name == "fasterrcnn":
        model = load_fasterrcnn(num_classes)
    elif model_name == "retinanet":
        model = load_retinanet(num_classes)
        lr = lr * 0.1
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    # transfer model to GPU
    torch.backends.cudnn.benchmark = True
    model.to(device)
    
    # Create data loaders 
    train_loader = create_dataloader("train", dataset_path, True, size, batch, workers)
    val_loader = create_dataloader("validation", dataset_path, False, size, batch, workers)
    test_loader = create_dataloader("test", dataset_path, False, size, batch, workers)

    # After this line in train_torch_detector:
    train_loader = create_dataloader("train", dataset_path, True, size, batch, workers)
    val_loader = create_dataloader("validation", dataset_path, False, size, batch, workers)
    test_loader = create_dataloader("test", dataset_path, False, size, batch, workers)

    # Optimiser
    optimizer = get_optimizer(model, lr, momentum, weight_decay, optimizer)

    # Determine best model by AP
    best_map = -float("inf")
    # Early stopping check
    patience = patience # No improvement checks in val
    num_no_improve = 0 # Num of consec checks without improvement
    scaler = GradScaler("cuda")

    # Top of metrics per epoch in metrics file
    with open(metrics_path, "a+", encoding="utf-8") as f:
        f.write("\n\nepoch | train_loss | val_loss | mAP_0.5 | mAP_0.5:0.95\n")

    # Loop through each epoch
    for epoch in range(epochs):
        print(f"\n[{model_name}] Epoch {epoch + 1}/{epochs}")
        map_05_095 = None
        val_loss = None

        # Training
        train_loss = train_epoch(model, optimizer, train_loader, device, epoch, scaler)
        # Validation on every 5 and last epoch
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            val_loss, map_results = validation_epoch(model, val_loader, device, epoch, scaler)
            map_05_095 = float(map_results["map"])

            with open(metrics_path, "a+", encoding="utf-8") as f:
                f.write(
                    f"{epoch + 1:5d} | "
                    f"{train_loss:0.4f} | "
                    f"{val_loss:0.4f} | "
                    f"{map_results['map_50']:0.4f} | "
                    f"{map_results['map']:0.4f}\n"
                )

        # Early stopping if no improvement for patience
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            if map_05_095 is not None:
                if map_05_095 > best_map: # Save best model weights
                    best_map = map_05_095
                    torch.save(model.state_dict(), best_model)
                    num_no_improve = 0
                    print(f"Best new mAP_0.5:0.95 = {best_map:.4f} at epoch {epoch + 1}")
                else:
                    num_no_improve += 1

                # Check if early stopping
                if num_no_improve >= patience:
                    print("Early stopping...")
                    break
    print("Training completed.")
    best_model = rf"model_weights\best\{model_name}.pt"
    # Test metrics
    final_metrics(model, device, size, str(best_model), str(metrics_path), 
                  val_loader, test_loader, None, batch, model_name=model_name)


#__________ MAIN PIPELINE __________

def run_pipeline(prepare, augmentation, run_yolo,  run_torch, yolo_model, torch_models, 
                 epochs, size, yolo_batch, torch_batch, patience, optimizer, 
                 lr0, lrf, weight_decay, momentum, cos_lr, cache, workers):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Resplitting and TXT label creation
    if prepare:
        prepare_dataset(dataset_path=dataset_path)

    # Run YOLO
    if run_yolo:
        train_yolo(
            augmentation=augmentation,
            model_name=yolo_model,
            epochs=epochs,
            size=size,
            batch=yolo_batch,
            patience=patience,
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            weight_decay=weight_decay,
            momentum=momentum,
            cos_lr=cos_lr,
            cache=cache,
            workers=workers,
            device=device,
        )

    # Run defined torch models
    if run_torch:
        for model_name in torch_models:
            train_torch_detector(
                model_name=model_name,
                epochs=epochs,
                size=size,
                batch=torch_batch,
                patience=patience,
                optimizer=optimizer,
                lr=lr0,
                weight_decay=weight_decay,
                momentum=momentum,
                device=device, 
                workers=workers
            )

# Training parameters
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--augmentation", action="store_true")
    # Model Running
    parser.add_argument("--run_yolo", action="store_true")
    parser.add_argument("--run_torch", action="store_true")
    # YOLO Model
    parser.add_argument("--yolo_model", default="yolo11n")
    parser.add_argument("--torch_models", nargs="+", default=["fasterrcnn"])
    # Training configuration
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--yolo_batch", type=int, default=12)
    parser.add_argument("--torch_batch", type=int, default=2)
    # Patience 
    parser.add_argument("--patience", type=int, default=30)
    # Optimizer
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--cos_lr", action="store_true")
    # System 
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    # Get arguments
    args = parser.parse_args()

    run_pipeline(
        # Dataset
        prepare=args.prepare,
        augmentation=args.augmentation,
        # Models
        run_yolo=args.run_yolo,
        run_torch=args.run_torch,
        yolo_model=args.yolo_model,
        torch_models=args.torch_models,
        # Training Configs
        epochs=args.epochs,
        size=args.size,
        yolo_batch=args.yolo_batch,
        torch_batch=args.torch_batch,
        patience=args.patience,
        # Optimiser
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        cos_lr=args.cos_lr,
        cache=args.cache,
        workers=args.workers
    )