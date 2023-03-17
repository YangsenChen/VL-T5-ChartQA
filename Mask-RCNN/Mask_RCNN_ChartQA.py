# !nvidia-smi
#
#
#
# # Download your Sample Data
#
# # !unzip -q data.zip
#
# !pip install 'git+https://github.com/facebookresearch/detectron2.git'
#
# !python train.py --train-data-dict data/train_dict.pickle --valid-data-dict data/val_dict.pickle --train-images /mnt/c/Users/yangsen/Desktop/intern/ChartQA/ChartQA Dataset/train/png --valid-images data/images_validation/ --output-dir ../data/train/features/ --ITERS 16000 --batch-size 2


import json
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

import torch
from tqdm import tqdm
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Change these paths to your model, images, and saving paths
trained_model_path = "maskrcnn.pth"
images_path = "/mnt/c/Users/yangsen/Desktop/intern/ChartQA/ChartQA Dataset/train/png/"
save_path = "../data/train/features/"
CLASSES_NUM = 15



from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml")  # Let training initialize from model zoo
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = CLASSES_NUM  # number of classes
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

import detectron2.data.transforms as T
from detectron2.structures import ImageList
aug = T.ResizeShortestEdge(
    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
)
input_format = cfg.INPUT.FORMAT

print(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, trained_model_path)  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
model = build_model(cfg)
DetectionCheckpointer(model).load(trained_model_path)
model.eval()

images_names = os.listdir(images_path)
images_paths = [images_path + x for x in os.listdir(images_path)]

# Generate Visual Features for each image
for image_name in tqdm(images_names):   
    my_file = Path(save_path+str(image_name.split(".")[0])+'.json')
    # Pass if the file has been processed before
    if my_file.is_file():
        continue
    
    orig_image = cv2.imread(images_path+image_name)
    orig_height, orig_width = orig_image.shape[:2]
    image = aug.get_transform(orig_image).apply_image(orig_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    c, height, width = image.shape
    inputs = [{"image": image, "height": height, "width": width}]
    #run model
    image_data_dict = {}

    with torch.no_grad():
        images = model.preprocess_image(inputs) 
        features = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, features) 
        #### orig
        instances, _ = model.roi_heads(images, features, proposals)
        mask_features = [features[f] for f in model.roi_heads.in_features]
        mask_features = model.roi_heads.pooler(mask_features, [x.pred_boxes for x in instances])
        mask_features = model.roi_heads.res5(mask_features)
        average_pooling = nn.AvgPool2d(7)
        
        mask_features = average_pooling(mask_features)
        orig_size = mask_features.size()
        mask_features = mask_features.resize_(orig_size[0], orig_size[1])
        ###

        boxes = instances[0].pred_boxes.tensor.detach().cpu().numpy()

        normalized_boxes = [[box[0] / width, box[1] / height, box[2] / width, box[3] / height] for box in boxes]
        image_data_dict['visual_feats'] = mask_features.detach().cpu().numpy().tolist()
        image_data_dict['bboxes'] = normalized_boxes

    with open(save_path+str(image_name.split(".")[0])+'.json', 'w') as f:
       json.dump(image_data_dict, f)


