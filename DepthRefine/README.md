# DeepDepthRefiner
Depth map refine using Pixel-wise Occlusion Orientation Relation Map (P2ORM)

## Training on InteriorNet_OR 

Our model is trained on 10,160 images of InteriorNet for 30 epochs 
using batch size of 8 and Adam optimizer with learning rate of 1e-5:
```
python train.py
```
The whole training procedure can be finished in one day with a single TitanX GPU.

Trained model weights would be saved in ``#save_dir/ckpt.pth``

## Testing on iBims1_OR

For this dataset, we conduct the depth refine and evaluation at the same time for each method.

To refine the depth map originally predicted by some Monocular Depth Estimation method with 
our depth refine model:
```
python refine_ibims.py --checkpoint #save_dir/ckpt.pth --depth_pred_method #method_name
```
This cmd would refine the depth map predicted by **#method_name**, which should be chosen from: 
\[eigen, fayao, fcrn, junli, planenet, sharpnet\].


## Testing on NYUv2_OR

