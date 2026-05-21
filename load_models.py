import torch
# SSD300 VGG16
import torchvision
from torchvision.models.detection import _utils
from torchvision.models.detection import SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
# Faster R-CNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
# RetinaNet
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

''' Retrieve the reconfigured Torchvision detector models '''

#__________ SSD __________

def load_ssd(num_classes, size):
    # Load the Torchvision pretrained model.
    model = torchvision.models.detection.ssd300_vgg16(
        weights=SSD300_VGG16_Weights.COCO_V1
    )
    # Retrieve the backbone output channel sizes for SSD head input
    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
    # List containing number of anchors based on aspect ratios.
    num_anchors = model.anchor_generator.num_anchors_per_location()
    # Configuring the classification head 
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )
    # Resize input image to expected (if required)
    model.transform.min_size = (size,)
    model.transform.max_size = size
    
    return model

#__________ FASTER R-CNN __________

def load_fasterrcnn(num_classes):
    # Load pre-trained Faster R-CNN model with ResNet50 backbone and FPN
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    # Get the number of input features for the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace classifier head (to dataset's num of classes)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

#__________ RETINANET __________

def load_retinanet(num_classes):
    # Load pre-trained RetinaNet model with ResNet50 backbone and FPN
    weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
    model = retinanet_resnet50_fpn(weights=weights)
    # Retrieve the backbone output channel sizes for RetinaNet head input
    in_channels = model.backbone.out_channels
    # List of the number of anchors
    num_anchors = model.head.classification_head.num_anchors
    # Replace classification head
    model.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
    
    return model

#__________ OPTIMISER __________

def get_optimizer(model, lr=0.001, momentum=0.9, weight_decay=0.0005, opt_name="AdamW"):
    # Trainable params
    params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )
    else: # Default SGD
        optimizer = torch.optim.SGD(
            params, 
            lr = lr, 
            momentum = momentum, 
            weight_decay = weight_decay
        )     
    return optimizer
