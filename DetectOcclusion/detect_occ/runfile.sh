#!/usr/bin/env bash
# train from scratch on BSDSownership dataset
python train_val.py --config ../experiments/configs/BSDSownership_order_myUnet_CCE_1.yaml --gpus 1

# test from trained model on BSDSownership dataset
python train_val.py --config ../experiments/configs/BSDSownership_order_myUnet_CCE_1.yaml --evaluate --resume 2020-01-15_15-40-52/checkpoint_19.pth.tar --gpus 0



