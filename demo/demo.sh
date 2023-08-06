export task=semantic
python demo.py --config-file ../configs/mapillary_vistas/dinat/oneformer_dinat_large_bs16_300k.yaml \
  --input /mnt/source/datasets/img_230607/* \
  --output /mnt/source/datasets/shibuya_oneformer/ \
  --task $task \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS /mnt/source/OneFormer/250_16_dinat_l_oneformer_mapillary_300k.pth