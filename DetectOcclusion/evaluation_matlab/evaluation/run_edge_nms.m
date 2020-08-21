% script to do edge detection nms 
dataset = 'DenseDepthNYUDv2';
res_root_dir = '/home/xuchong/Projects/occ_edge_order/experiments/output/2020-01-15_23-30-16/results_vis/test_74_DenseDepthNYUDv2_train/images';

switch dataset
    case 'ibims'
        testIdsFilename = fullfile(pwd, '..', '..', '/data/dataset_real/ibims/test_ori_iids.txt');
    case 'nyuv2'
        testIdsFilename = fullfile(pwd, '..', '..', '/data/dataset_real/NYUv2/test_iids.txt');
    case 'DenseDepthNYUDv2'
        testIdsFilename = fullfile(pwd, '..', '..', '/data/DenseDepthNYUDv2/data/test_iids_DenseDepthNYUDv2Train.txt');
end

ImageList = textread(testIdsFilename, '%s');

for ires = 1:length(ImageList)
        prob_in_path  = fullfile(res_root_dir, [ImageList{ires}, '_lab_v_g.png']);  
        prob_out_path = fullfile(res_root_dir, [ImageList{ires}, '_lab_v_g_nms.png']);  
        disp(prob_in_path)   
     
        % read imgs
        edge_prob = imread(prob_in_path);  % uint8
        edge_prob = single(edge_prob) / 255;  % [0~1]
        
        % do nms
        edge_prob_thin = edge_nms(edge_prob, 0);
        
        % output imgs
        edge_prob_thin = cast(edge_prob_thin * 255., 'uint8');
        imwrite(edge_prob_thin, prob_out_path);
end


