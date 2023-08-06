




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





def get_data_dicts(directory, classes):
    dataset_dicts = []
    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}

        filename = os.path.join(directory, img_anns["imagePath"])

        record["file_name"] = filename
        record["height"] = 1080
        record["width"] = 1920

        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']] # x coord
            py = [a[1] for a in anno['points']] # y-coord
            poly = [(x, y) for x, y in zip(px, py)] # poly for segmentation
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(anno['label']),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

classes = ['BL', 'CL', 'DM', 'JB', 'LA', 'PC', 'RA', 'SA', 'SL', 'SLA', 'SRA']






for d in ["train", "test"]:
    DatasetCatalog.register(
        "ceymo_" + d,
        lambda d=d: get_data_dicts(data_path+d, classes)
    )
    MetadataCatalog.get("ceymo_" + d).set(thing_classes=classes)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")


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

#cfg = get_cfg()
microcontroller_metadata = MetadataCatalog.get("ceymo_train")

#cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.merge_from_file("configs/Cityscapes/mask_rcnn_R_50_FPN.yaml")

cfg.DATASETS.TRAIN = ("ceymo_train",)
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")  # Let training initialize from model zoo


#cfg.MODEL.WEIGHTS = os.path.join("/mnt/source/OneFormer/model_city_8_0.0025_1000_128_11.pth")
cfg.MODEL.WEIGHTS = os.path.join("/mnt/source/OneFormer/model_ceymo_4_128.pth")

#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1

data_path = '/mnt/source/datasets/CeyMo/'

cfg.DATASETS.TEST = ("ceymo_test", )
test_dataset_dicts = get_data_dicts(data_path+'test', classes)

predictor = DefaultPredictor(cfg)
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

classes = ['BL', 'CL', 'DM', 'JB', 'LA', 'PC', 'RA', 'SA', 'SL', 'SLA', 'SRA']


'''
for d in ["train", "test"]:
    DatasetCatalog.register(
        "ceymo_" + d,
        lambda d=d: get_data_dicts(data_path+d, classes)
    )
    MetadataCatalog.get("ceymo_" + d).set(thing_classes=classes)
'''

microcontroller_metadata = MetadataCatalog.get("ceymo_test")
#for d in random.sample(test_dataset_dicts, 3):
img = cv2.imread(data_path + 'test/147a.jpg')
outputs = predictor(img)
v = Visualizer(img[:, :, ::-1],
                metadata=microcontroller_metadata,
                scale=0.8,
                instance_mode=ColorMode.IMAGE_BW # removes the colors of unsegmented pixels
)
#print(outputs)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
image = cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
cv2.imwrite('/mnt/source/datasets/out.png', image)
#plt.figure(figsize = (14, 10))
#plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
#plt.show()