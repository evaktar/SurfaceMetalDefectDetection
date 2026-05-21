import os 
import torch
import glob
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor

''' Dataset Helper Functions '''

#__________ IMAGES AND LABELS __________

# Retrieve list of full path of labels and images
def get_images_labels_paths(dataset_path):
    images_path = os.path.join(dataset_path, "images")
    labels_path = os.path.join(dataset_path, "labels")
    image_labels_paths = []

    # Subfolders in images folder e.g., crazing/
    for class_folder in sorted(os.listdir(images_path)):
        # Get full subfolder path e.g., images/crazing
        class_img_dir = os.path.join(images_path, class_folder)
        if not os.path.isdir(class_img_dir): # Skip if not a folder
            continue

        # Add .jpg for full image path
        for img_path in glob.glob(os.path.join(class_img_dir, "*.jpg")):
            # File name without extension (crazing_1)
            stem = os.path.splitext(os.path.basename(img_path))[0]
            # Find corresponding label in labels folder per class
            label_path = os.path.join(labels_path, class_folder, f"{stem}.txt")
            if os.path.isfile(label_path):
                # Add matching image and label paths
                image_labels_paths.append((img_path, label_path))
            else:
                print("Label not found for:", img_path)

    return image_labels_paths

#__________ CUSTOM TORCH DATALOADER __________

# Custom Dataset for torch models 
class TorchDataset(Dataset):
    def __init__(self, dataset_path, transforms=ToTensor(), class_offset=1):
        self.dataset_path = dataset_path
        self.transforms = transforms
        self.class_offset = class_offset
        # Flat list of (image_path, label_path)
        self.images = get_images_labels_paths(dataset_path)

    def __len__(self):
        return len(self.images) # Num of images

    def __getitem__(self, idx):
        img_path, label_path = self.images[idx]
        # Load image
        image = Image.open(img_path).convert("RGB")
        # Apply transforms first (Resize + ToTensor)
        if self.transforms is not None:
            image = self.transforms(image)  # tensor: C x H x W
        _, h, w = image.shape  #  tensor shape
        
        boxes = []
        labels = []
        # Load labels
        if os.path.getsize(label_path) > 0: # If not empty
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    # YOLO bboxes to pixel coords + class ID
                    cls_id = int(parts[0])
                    x_c = float(parts[1]) * w
                    y_c = float(parts[2]) * h
                    bw  = float(parts[3]) * w
                    bh  = float(parts[4]) * h
                    # Format to corner bbox
                    xmin = x_c - bw / 2.0
                    ymin = y_c - bh / 2.0
                    xmax = x_c + bw / 2.0
                    ymax = y_c + bh / 2.0
                    # Torchvision format [xmin, ymin, xmax, ymax]
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(cls_id + self.class_offset) # 0 reserved for bg
        # Convert bounding boxes + labels to tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            # Empty tensors if no boxes
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        # No crowd labels
        iscrowd = torch.zeros((labels.shape[0],), dtype=torch.int64)
        # Image and ground truth info
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": idx,
            "area": area,
            "iscrowd": iscrowd,
        }
        return image, target

# Transform images
def get_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)), # Image resize
        transforms.ToTensor() # PIL to PyTorch tensor
    ])
# Collate function
def torch_collate_fn(batch):
    # Unpack batches [(image1, target1), (image2, target2)]
    return tuple(zip(*batch))  # to ([image1, image2], [target1, target2]) 

# Create dataloaders for models
def create_dataloader(split, dataset_path, shuffle, size, batch=2, num_workers=1):
    # Dataset split (train/test/validation) path and transform
    split_dataset = TorchDataset(
        dataset_path / split,
        transforms=get_transform(size)
    )
    collate_fn = torch_collate_fn
    
    # Return DataLoader
    loader = DataLoader(
        split_dataset, 
        batch_size=batch, 
        shuffle=shuffle, 
        collate_fn=collate_fn,
        pin_memory=True,  # Pinned host memory (faster)
        num_workers=num_workers
        )
    return loader 
