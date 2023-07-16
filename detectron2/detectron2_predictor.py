import torch
#assert torch.__version__.startswith("1.8") 
import torchvision
import cv2


import os
import numpy as np
import json
import random
import matplotlib.pyplot as plt

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

cfg = get_cfg()
microcontroller_metadata = MetadataCatalog.get("ceymo_train")

#cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.merge_from_file("configs/Cityscapes/mask_rcnn_R_50_FPN.yaml")

cfg.DATASETS.TRAIN = ("ceymo_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")  # Let training initialize from model zoo

cfg.SOLVER.IMS_PER_BATCH = 4 # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11 # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
predictor = DefaultPredictor(cfg)


'''
citysacpes_data = '/mnt/source/datasets/stuttgart_00/'
files = os.listdir(citysacpes_data)
for val in files:
    file_name = citysacpes_data + val
    img = cv2.imread(file_name)
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1],
                    metadata=microcontroller_metadata, 
                    scale=0.8, 
                    instance_mode=ColorMode.IMAGE_BW # removes the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    image = cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
    #plt.figure(figsize = (14, 10))
    #plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    #plt.savefig('/mnt/source/datasets/cityscapes_mark/'+val)
    cv2.imwrite('/mnt/source/datasets/cityscapes_mark/'+val, image)
    print(val)
'''
'''
file_name = stuttgart_00_000000_000037_rightImg8bit
img = cv2.imread(file_name)
outputs = predictor(img)
v = Visualizer(img[:, :, ::-1],
                metadata=microcontroller_metadata, 
                scale=0.8, 
                instance_mode=ColorMode.IMAGE_BW # removes the colors of unsegmented pixels
)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
image = cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
#plt.figure(figsize = (14, 10))
#plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
#plt.savefig('/mnt/source/datasets/cityscapes_mark/'+val)
cv2.imwrite('/mnt/source/datasets/'+val, image)
print(val)
'''