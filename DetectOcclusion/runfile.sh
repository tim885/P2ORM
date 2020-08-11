#!/usr/bin/env bash
# test from resumed model
python train_val.py --config ../experiments/configs/ibims_order_myUnet_MaskCCE_1.yaml --evaluate --resume 2019-08-01_19-40-46/model_best_196.pth.tar --gpus 0



