This project uses:
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 
for YOLO-based detectors.
- [PyTorch](https://pytorch.org/) for RetinaNet, SSD, and Faster R-CNN implementations.
- [Albumentations](https://github.com/albumentations-team/albumentations) for data augmentation.

Edited ultralytics files:
- ultralytics\nn\tasks.py: Added custom/ultralytics CBAM YAML args parsing
- ultralytics\nn\modules\__init__.py: Included custom CBAM and ultralytics CBAM reference
- ultralytics\nn\modules\block.py: Added custom CBAM block
- ultralytics\nn\modules\conv.py: Renamed ultralytics CBAM to CBAMDefault