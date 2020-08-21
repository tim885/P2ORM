if [[ "$1" != "" ]]; then
    gpu="$1"
else
    gpu=0
fi

label_dir='/home/xuchong/ssd/Projects/occ_edge_order/data/dataset_real/NYUv2/label/val_occ_order_raycasting_1mm_woNormal'

CUDA_VISIBLE_DEVICES=$gpu python train_val_nyu.py \
--occ_dir $label_dir --save_dir ckpt \
--epoch 30 --lr 1e-5 --batch_size 8 \
--alpha_occ 0.01 --use_occ --th 15

