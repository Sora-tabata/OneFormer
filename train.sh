python train_net.py --dist-url 'tcp://127.0.0.1:50163' \
    --num-gpus 1 \
    --config-file configs/ade20k/swin/oneformer_swin_large_bs16_160k.yaml \
    OUTPUT_DIR outputs/ade20k_swin_large WANDB.NAME ade20k_swin_large