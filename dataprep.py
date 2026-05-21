# Path Libraries
import os
import xml.etree.ElementTree as ET
import shutil 
from pathlib import Path
import albumentations as A

''' Data Preprocessing '''

#__________ TEXT ANNOTATIONS __________

# Convert Pascal VOC XML annotations to YOLO TXT format
def xmlToTxt(class_mapping, datasets):
    converted_count = 0
    # For each dataset partition
    for path in datasets:
        imagesDir = os.path.join(path, "images")
        xmlDir = os.path.join(path, "annotations")
        txtDir = os.path.join(path, "labels")
        # Retrieve class folders (from images)
        class_folders = []
        if os.path.exists(imagesDir):
            class_folders = [
                f
                for f in os.listdir(imagesDir)
                if os.path.isdir(os.path.join(imagesDir, f))
            ]
            # Create labels folder and subfolders if they do not exist
            os.makedirs(txtDir, exist_ok=True)
            for cls in class_folders:
                os.makedirs(os.path.join(txtDir, cls), exist_ok=True)

        # Iterate through the XML
        for xmlFile in os.listdir(xmlDir):
            if not xmlFile.endswith(".xml"):
                continue
            xmlPath = os.path.join(xmlDir, xmlFile)
            # Parse through the XML elements
            try:
                tree = ET.parse(xmlPath)
                root = tree.getroot() # Top element
            except Exception as e:
                print(f"Failed to parse {xmlFile}: {e}")
                continue
            # Find the image dimensions
            size = root.find("size")
            try: # Convert to int
                img_w = int(size.find("width").text)
                img_h = int(size.find("height").text)
            except Exception as e:
                print(f"{xmlFile}, invalid image size ({e})\n")
                continue

            bboxList = []
            object_classes = []
            primary_class = None # If multiple classes
            # Find defect objects 
            for obj in root.findall("object"):
                # Get class
                className = obj.find("name").text
                object_classes.append(className)
                # Set primary class to the object's class
                if primary_class is None:
                    primary_class = className
                classIdx = class_mapping[className]
                
                # Find bounding boxes
                bbox = obj.find("bndbox")
                if bbox is None:
                    print(f"Object without bndbox")
                    continue
                try:
                    # Retrieve Pascal VOC style coordinates
                    xmin = int(bbox.find("xmin").text)
                    ymin = int(bbox.find("ymin").text)
                    xmax = int(bbox.find("xmax").text)
                    ymax = int(bbox.find("ymax").text)
                except Exception as e:
                    print(f"  [WARNING] {xmlFile}: invalid bbox ({e})")
                    continue

                # Convert to normalised YOLO format 
                x_center = ((xmin + xmax) / 2) / img_w
                y_center = ((ymin + ymax) / 2) / img_h
                width = (xmax - xmin) / img_w
                height = (ymax - ymin) / img_h
                # (x_center, y_center, width, height)
                bboxList.append(
                    f"{classIdx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

            # Create directory if it does not exist
            target_dir = os.path.join(txtDir, primary_class)
            os.makedirs(target_dir, exist_ok=True)
            # Create text label
            txtFile = os.path.join(
                target_dir, xmlFile.replace(".xml", ".txt")
            )
            # Write created text files 
            if os.path.exists(txtFile):
                continue
            else:
                with open(txtFile, "w") as f:
                    f.write("\n".join(bboxList))
                converted_count += 1
                print(f"Wrote {txtFile} \n")

    # State how many files were converted
    if converted_count == 0:
        print("\nNo conversion needed.\n")
    else:
        print(f"\nConverted {converted_count} XML files to YOLO TXT.\n")

#__________ RESPLIT DATASET __________

# Count images per class
def per_class_count_images(folder_path, class_folders):
    class_totals = {} # dictionary of counts per class

    # Loop through list of subfolders (classes)
    for folder_name in class_folders:
        folder = folder_path / folder_name
        if not folder.exists():
            print(f"{folder_name}: Folder not found")
            continue
        count = sum(
            1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() == '.jpg'
        ) # Sum of images in subfolder
        class_totals[folder_name] = count # add to totals dict

    return class_totals

# Check if dataset partitions are at ratio 80:10;10
def check_split(folder_paths, class_folders):
    all_counts = {}

    for folder_path in folder_paths:
        split = folder_path.parts[-2] # e.g. train from ../train/images
        counts = per_class_count_images(folder_path, class_folders)
        all_counts[split] = counts # counts per split

    totals = {split: sum(class_counts.values()) for split, class_counts in all_counts.items()}
    #totals = {'train': 1440, 'validation': 180, 'test': 180}

    total = sum(totals.values()) # Overall dataset images

    for split, class_count in totals.items():
            expected = round(total * (0.80 if split == "train" else 0.10))
            if class_count != expected: # if a count is not expected return false
                return False

    return True

# Resplit the dataset if partitions are not at ratio 80:10:10
def data_resplit(folders, folder_paths, class_folders, validation_path, test_path, images_count):

    if not check_split(folder_paths, class_folders): # Check if needs split
        # Validation folder paths e.g. NEU-DET/validation/image
        val_images = validation_path / folders[0]
        val_labels = validation_path / folders[1]
        val_ann = validation_path / folders[2]
        # Test folder paths
        test_images = test_path / folders[0]
        test_labels = test_path / folders[1]
        test_ann = test_path / folders[2]

        # Create test directories if do not exist
        for d in [test_images, test_labels, test_ann]:
            d.mkdir(parents=True, exist_ok=True)

        # For each class in folder
        for class_name in class_folders:
            print(f"\nProcessing class: {class_name}")
            # Validation paths for each class e.g. NEU-DET/validation/image/crazing
            val_img_dir = val_images / class_name
            val_lab_dir = val_labels / class_name
            val_xml_dir = val_ann
            # Test paths for each class
            test_img_dir = test_images / class_name
            test_lab_dir = test_labels / class_name
            # Create class subfolder if does not exist
            test_img_dir.mkdir(exist_ok=True)
            test_lab_dir.mkdir(exist_ok=True)

            # Get all image files from val, sort by name
            img_files = [
                p for p in val_img_dir.iterdir()
                if p.is_file() and p.suffix.lower() == '.jpg'
            ]
            if len(img_files) < images_count: # If number of images < images to be moved
                print(f"Not enough images in {class_name}: {len(img_files)}")
                continue
            # Last (sorted) images 
            imgs_to_move = sorted(img_files)[-images_count:] 
            
            # For each image to be moved
            for img_path in imgs_to_move:
                stem = img_path.stem  # e.g. "crazing_241"
                # Find TXT label for image
                lab_path = val_lab_dir / f"{stem}.txt"
                if not lab_path.exists():
                    print(f"Missing label for {img_path.name}, skipping.")
                    continue
                # Find XML annotation for image
                xml_path = val_xml_dir / f"{stem}.xml"
                if not xml_path.exists():
                    print(f"Missing annotation for {img_path.name}, skipping.")
                    continue
                # Move to test directories
                shutil.move(img_path, test_img_dir / img_path.name)
                shutil.move(lab_path, test_lab_dir / lab_path.name)
                shutil.move(xml_path, test_ann  / xml_path.name)
                print(f"Moved {img_path.name} to test folder.")
    else:
        print("Dataset has already been split to 80:10:10 train/val/test")

#__________ AUGMENTATION __________

# Augmentation techniques
def get_custom_aug():
    return [
        # Lighting conditions
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1),
            contrast_limit=(-0.1, 0.1),
            p=0.3,
        ),
        # Orientation invariance
        A.HorizontalFlip(p=0.3),
        # Add noise (industrial settings)
        A.GaussNoise(std_range=(0.01, 0.02), p=0.05), 
        # Colour shift (camera)
        A.HueSaturationValue(
            hue_shift_limit=(-10, 10),
            sat_shift_limit=(-30, 30),
            val_shift_limit=(-20, 20),
            p=0.4,
        )
    ]

#__________ OVERSAMPLING __________

# Oversample the weaker classes
def oversample(images_root, labels_root, output_path, target_class=(0, 4), duplicate=3):
    images_root = Path(images_root)
    labels_root = Path(labels_root)
    output_path = Path(output_path)
    # Store duplicated image paths
    lines = []

    # Train/images folder
    for img_path in images_root.rglob("*.jpg"):
        # Get matching label (same subfolder)
        rel = img_path.relative_to(images_root)
        label_path = labels_root / rel.with_suffix(".txt")
        if not label_path.exists():
            continue
        # Get Class IDs from labels files
        with open(label_path, "r", encoding="utf-8") as f:
            cls_ids = [int(line.split()[0]) for line in f if line.strip()]
        
        # 'Duplicate' the image if classes are in that image
        if any(cid in target_class for cid in cls_ids):
            for _ in range(duplicate): # by x times
                lines.append(str(img_path))
        else:
            lines.append(str(img_path))

    # Write to NEU-DET folder
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
