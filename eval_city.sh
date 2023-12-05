export DETECTRON2_DATASETS="/mnt/source/datasets"
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:24"
#python3 oneformer/data/datasets/register_mapillary_vistas_panoptic.py

python train_net.py --dist-url 'tcp://127.0.0.1:50164' \
    --num-gpus 1 \
    --config-file configs/cityscapes/convnext/mapillary_pretrain_oneformer_convnext_xlarge_bs16_90k.yaml \
    --eval-only MODEL.IS_TRAIN False MODEL.TEST.TASK panoptic
