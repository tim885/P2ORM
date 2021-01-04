This is the matlab script code for oriented occlusion boundary detection task on BSDSownership/NYUv2-OR/iBims-1 dataset and tests successfuly on Matlab2016b. You can use it to evaluate and visulize the oriented occlusion boundary detection results. The code is originally developed by Guoxia Wang and Peng Wang and offered in https://github.com/GuoxiaWang/DOOBNet/tree/master/doobscripts. 

# installation

Unzip the evaluation_code.zip in this directory.  

# Evaluation

There are two main interfaces:
*  `EvaluateOcc.m`
   This script perform the evaluation.
   In this script, you need to set following variables:
   ```
   oriGtPath          it specifies the directory that contains mat files of test images. 
   oriResPath         it specifies the root directory of prediction results.
   testIdsFilename    the relative or absolute path of test ids file. 
   opt.method_folder  the folder name in oriResPath. 
   opt.model_name     the folder name in opt.method_folder that contains mat files of your methods prediction.
   ...
   ...                you can see the other variables in EvaluateOcc.m
   ```
   ```shell
   # evaluate on metrics
   cd evaluation/
   matlab -nodisplay
   run EvaluateOcc.m
   ```
   The orientation occlusion estimation metrics are showed under 'Orientation PR'. 
   
   The occlusion probability maps after NMS are stored in dir opt.method_folder/test_epoch_dataset/images/.

*  `OccCompCurvesPlot.m`
   This script plots multi algorithms occlusion boundary Precision-Recall curves, oriented occlusion boundary Precision-Recall curves, oriented occlusion boundary Accuracy-Recall curves.
   In this script, you need to set following vairables:
   ```
   algs        N x 1 cell, each element is pridiction path
   nms         N x 1 cell, each element is algorithm name which will be present at legend
   export_dir  the directory that you want to save the plots
   ```
   ```shell
   # plot performance curves
   cd evaluation/
   matlab -nodisplay 
   run OccCompCurvesPlot.m
   ```
   Figures are stored in export_dir in the code. To plot comparision curves on BSDSownership, download results of other methods [here](https://1drv.ms/u/s!AhUxUphMG7I4jZNaP0oxM8MqJImW9Q?e=vwhcfP).
   
   N.B. It's recommended to run under matlab command mode (matlab -nodisplay) to avoid possible matlab crash. 

# Visualization

The visualization script supports 3 type figures: original image, ground truth oriented occlusion boundary map with arrow and corresponding prediction map.

*  `PlotAll.m` is the main point. You need to set following variables:
   ```
   img_dir            the original images directory. 
   gt_mat_dir         the ground truth annotation directory. 
   res_mat_dir        the prediction directory, which contans .mat files
   export_dir         the directory that you want to save the print eps figure
   test_ids_filename  the relative or absolute path of test ids file. 
   ```
   ```shell 
   # plot figures
   cd visualization/ 
   matlab -nodisplay
   run PlotAll.m.
   ``` 
   Figures are stored in export_dir in the code.
   
   N.B. It's recommended to run under matlab command mode (matlab -nodisplay) to avoid possible matlab crash. 


# Issues

*  If you meet the error about edgesNmsMex or correspondPixels, you can download [edge box](https://github.com/pdollar/edges) and add the path to matlab path or copy edgesNmsMex.mexa64 and correspondPixels.mexa64 from `edges/private` folder to `common` folder. If you can not run, please recompile the edgesNmsMex and correspondPixels according the README.
*  If you meet the error about convConst or gradientMex, you can download [piotr's toolbox](https://github.com/pdollar/toolbox) and add the path to matlab path or copy convConst.mexa64 and gradientMex.mexa64 from `piotr_toolbox/channels/private` folder to `common` folder. If you can not run, please get more information from author's website.
