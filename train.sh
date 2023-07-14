export DETECTRON2_DATASETS="/mnt/source/datasets/mapillary_vistas/"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:24"
python3 oneformer/data/datasets/register_mapillary_vistas_panoptic.py

python train_net.py --dist-url 'tcp://127.0.0.1:50163' \
    --num-gpus 1 \
    --num-machines 1 \
    --machine-rank 0 \
    --config-file configs/mapillary_vistas/dinat/oneformer_dinat_large_bs16_300k.yaml \
    OUTPUT_DIR outputs/mapillary WANDB.NAME mapillary