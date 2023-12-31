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

cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
#cfg.merge_from_file("configs/Cityscapes/mask_rcnn_R_50_FPN.yaml")
#cfg.merge_from_file("configs/Cityscapes/mask_rcnn_R_50_FPN.yaml")

cfg.DATASETS.TRAIN = ("ceymo_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")  # Let training initialize from model zoo

cfg.SOLVER.IMS_PER_BATCH = 14# This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11 # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

#cfg.MODEL.WEIGHTS = os.path.join("/mnt/source/OneFormer/model_ceymo_coco_4_0.0025_1000_128_11.pth")
cfg.MODEL.WEIGHTS = os.path.join("/mnt/source/OneFormer/model_14.pth")
#cfg.MODEL.WEIGHTS = os.path.join("/mnt/source/OneFormer/model_city_8_0.001_1000_128_11.pth")

#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
predictor = DefaultPredictor(cfg)


'''
citysacpes_data = '/mnt/source/datasets/cityscapes_oneformer/'
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
    cv2.imwrite('/mnt/source/datasets/cityscapes_oneformer_mark_3/'+val, image)
    print(val)

'''
'''
def rename_class(pred_classes):
    torch.where(pred_classes == 0, torch.tensor('Bus Lane'), torch.tensor(False)
    torch.where(pred_classes == 1, torch.tensor('Cycle Lane'), torch.tensor(False)
    torch.where(pred_classes == 2, torch.tensor('Diamond'), torch.tensor(False)
    torch.where(pred_classes == 3, torch.tensor('Junction Box'), torch.tensor(False)
    torch.where(pred_classes == 4, torch.tensor('Right Arrow'), torch.tensor(False)
    torch.where(pred_classes == 5, torch.tensor('Pedestrian Crossing'), torch.tensor(False)
    torch.where(pred_classes == 6, torch.tensor('Left Arrow'), torch.tensor(False)
    torch.where(pred_classes == 7, torch.tensor('Straight Arrow'), torch.tensor(False)
    torch.where(pred_classes == 8, torch.tensor('Slow'), torch.tensor(False)
    torch.where(pred_classes == 9, torch.tensor('Straight-Right Arrow'), torch.tensor(False)
    torch.where(pred_classes == 10, torch.tensor('Straight-Left Arrow', torch.tensor(False)
''' 

citysacpes_data = '/mnt/source/datasets/cityscapes_oneformer/'
#citysacpes_data = '/mnt/source/datasets/CeyMo/test/'
citysacpes_data_ = '/mnt/source/datasets/stuttgart_00/'
citysacpes_data_ = '/mnt/source/datasets/shibuya_oneformer/'
file_name = 'stuttgart_00_000000_000360_rightImg8bit.png'
file_name = '1353.jpg'
img = cv2.imread(citysacpes_data_ + file_name)
outputs = predictor(img)
sorted_scores, idx = torch.sort(outputs['instances'].scores,descending=True)
outputs['instances'].pred_boxes = outputs['instances'].pred_boxes[idx]
outputs['instances'].scores = outputs['instances'].scores[idx]
outputs['instances'].pred_classes = outputs['instances'].pred_classes[idx]
outputs['instances'].pred_masks = outputs['instances'].pred_masks[idx]

print(outputs["instances"][0].pred_classes)
img_ = cv2.imread(citysacpes_data + file_name)
#rename_class(outputs['instances'].pred_classes)
v = Visualizer(img_[:, :, ::-1],
                metadata=microcontroller_metadata, 
                scale=0.8, 
                instance_mode=ColorMode.IMAGE # removes the colors of unsegmented pixels
)
v = v.draw_instance_predictions(outputs["instances"][[0, 1]].to("cpu"))
image = cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
#plt.figure(figsize = (14, 10))
#plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
#plt.savefig('/mnt/source/datasets/cityscapes_mark/'+val)
cv2.imwrite('/mnt/source/datasets/out.png', image)

#print(outputs['instances'].scores)
#print(outputs['instances'].pred_classes == 7)
#print(outputs['instances'].pred_masks[0].shape)