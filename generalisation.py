# Import libraries
import os
from ultralytics import YOLO

''' Generalisation of best NEU-DET trained 
    model on GC10-DET (zero-shot) cross-dataset'''

#__________ CONFIG __________  

gc10_path = "GC10-DET"
model_path = "runs/detect/yolo11n_aug_oversample/weights/best.pt"
original_map_50 = 0.774 # mAP@0.5 on NEU-DET
yaml_file = "config/gc10.yaml"
metrics_path = "metrics/generalisation.txt"
splits = ["train", "val", "test"]

#__________ CONVERT TO SINGLE DEFECT CLASS __________  

# Convert 10 defects to 1 defect class
def collapse_classes(base_path):
    # For each dataset partition
    for split in splits:
        # Get label path
        label_dir = os.path.join(base_path, split, "labels")
        if not os.path.exists(label_dir):
            print(f"Missing folder: {label_dir}")
            continue
        
        # Build list of label files
        files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

        # Traverse through lines in each file
        for file in files:
            file_path = os.path.join(label_dir, file)
            new_lines = []
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    # class x_center y_center width height
                    parts = line.strip().split()
                    # Replace classes with '0'
                    parts[0] = "0"
                    new_lines.append(" ".join(parts))
            # Overwrite with the new class
            with open(file_path, "w") as f:
                f.write("\n".join(new_lines))

#__________ EVALUATION __________  

# Evaluate best model
def evaluate_model(model_path, yaml_path):

    # Model validation
    model = YOLO(model_path)
    results = model.val(
        data=yaml_path,
        split="test",
        seed=42,
        imgsz=640,
        conf=0.001,
        iou=0.5,
        device=0,
        verbose=True
    )
    return results


#__________ METRICS __________  

def metrics(results):
    # Retrieve metrics
    metrics = results.box
    map_50 = metrics.map50
    map_5095 = metrics.map
    precision = metrics.mp
    recall = metrics.mr
    # F1 Score
    f1 = 2 * precision * recall / (precision + recall)

    drop = original_map_50 - map_50
    with open(metrics_path, "a+", encoding="utf-8") as f:
        f.write("\GC10 Generalisation\n")
        f.write(f"mAP@0.5: {map_50:.4f}\n")
        f.write(f"mAP@0.5:0.95 : {map_5095:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"\nGeneralisation Drop : {drop:.4f}\n")


#__________ MAIN __________  

if __name__ == "__main__":
    # Collapse GC10-DET classes to 1 defect class
    collapse_classes(gc10_path)
    # Retrieve evaluation metrics
    metrics(evaluate_model(model_path, yaml_file))