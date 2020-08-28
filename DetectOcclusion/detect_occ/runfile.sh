#!/usr/bin/env bash
# train from scratch for BSDSownership dataset
python train_val.py --config ../experiments/configs/BSDSownership_order_myUnet_CCE_1.yaml --gpus 1

# test from trained model for BSDSownership dataset
python train_val.py --config ../experiments/configs/BSDSownership_order_myUnet_CCE_1.yaml --evaluate --resume 2020-01-15_15-40-52/checkpoint_19.pth.tar --gpus 0

# train from scratch for iBims1_OR dataset
python train_val.py --config ../experiments/configs/ibims_order_myUnet_CCE_1.yaml --gpus 1

# test from trained model for iBims1_OR dataset
python train_val.py --config ../experiments/configs/ibims_order_myUnet_CCE_1.yaml --evaluate --resume iBims1OR_pretrained/iBims1OR_epoch179.pth.tar --gpus 0

# train from scratch for NYUv2_OR dataset
python train_val.py --config ../experiments/configs/nyuv2_order_myUnet_CCE_1.yaml --gpus 1

# test from trained model for NYUv2_OR dataset
python train_val.py --config ../experiments/configs/nyuv2_order_myUnet_CCE_1.yaml --evaluate --resume NYUv2OR_pretrained/NYUv2OR_epoch74.pth.tar --gpus 0


