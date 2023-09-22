export task=panoptic
python demo.py --config-file ../configs/cityscapes/convnext/mapillary_pretrain_oneformer_convnext_xlarge_bs16_90k.yaml \
  --input /mnt/source/datasets/img_230607/* \
  --output /mnt/source/datasets/shibuya_oneformer/ \
  --task $task \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS /mnt/source/OneFormer/model/model_ceymo230922.pth