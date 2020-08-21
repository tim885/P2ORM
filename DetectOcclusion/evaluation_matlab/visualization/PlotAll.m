% plot each sample with relevant visualizations. Currently only BSDSownership is supported.  
addpath('../export_fig');
addpath('../common');
addpath('edge_link');

DataSet = 'BSDSownership';
exp_name = 'BSDSownership_pretrained';
method_folder = [exp_name '/results_vis'];
which_epoch = '19';

disp('eval epoch: ');
disp(which_epoch); 
res_in_dir  = fullfile(pwd, '..', '..', 'experiments', 'output', method_folder, strcat('test_', num2str(which_epoch), '_', DataSet),'res_mat');
export_dir = fullfile(pwd, '..', '..', 'experiments', 'output', method_folder, strcat('test_', num2str(which_epoch), '_', DataSet),'res_viz');
    
img_dir = fullfile(pwd, '..', '..', 'data/BSDS300/images/train/');
gt_mat_dir = fullfile(pwd, '..', '..', 'data/BSDS300/BSDS_theta/testOri_mat/');
res_mat_dir = fullfile(res_in_dir, 'test_pred/');
test_ids_filename = fullfile(pwd, '..', '..', '/data/BSDS300/BSDS_theta/test_ori_iids.txt');         

opt.res_mat_dir = res_mat_dir;
opt.export_dir = export_dir;
opt.nms_thresh = 0.5;
opt.append = ''; 
opt.w_occ = 1; 
if opt.w_occ; opt.append = '_occ'; end 
opt.nthresh = 33; 
opt.thinpb = 1;  % whether use nms 
opt.print = 1;  % save figures
opt.validate = 0; 
opt.save_fig = 0; 
opt.curr_imgId = 0;

imgids = textread(test_ids_filename, '%s');

if opt.validate
    valid_num = 1; 
    imgids = imgids(1:valid_num); 
end 

nImgs = length(imgids);
mkdir(opt.export_dir);
for idx = 1:nImgs
    close all;

    I = imread([img_dir imgids{idx} '.jpg']);
    im_sz = size(I);
    I_white = uint8(ones(im_sz(1), im_sz(2), im_sz(3)) * 255);
    
    load([gt_mat_dir imgids{idx} '.mat'], 'gtStruct');
    gt_img = cat(3, gtStruct.gt_theta{:});
    
    % BSDS ownership gt annotation from two anntator. 3, 4 channel is the second 
    % annotation, you can change to 1, 2 if you want to plot the first annatation
    gt_edge = gt_img(:,:,3);
    gt_ori = gt_img(:,:,4);

    % original image
    figure(), imshow(I, 'InitialMagnification',100);
    truesize(gcf);
    saveas(gcf, sprintf('%s/%s_img.png', export_dir, imgids{idx}), 'png');
    close(gcf);

    % GT orientation w/ arrow
    PlotOcclusionArraw(I, gt_edge, gt_ori);
    if opt.print, 
        saveas(gcf, sprintf('%s/%s_occ_gt.png', export_dir, imgids{idx}), 'png');
        close(gcf);
    end

    % GT orientation w/ value heatmap
%     PlotOri(gt_ori);
%     if opt.print, export_fig(sprintf('%s/%s_ori_gt', export_dir, imgids{idx}), '-q101', '-transparent', '-depsc3', '-png', '-tight');  end;
    
    % GT boundary
%     figure, imshow(1-gt_edge, 'Border','tight'), truesize(gcf);
%     if opt.print, export_fig(sprintf('%s/%s_edge_gt', export_dir, imgids{idx}), '-q101', '-transparent', '-depsc3', '-png', '-tight');  end;
    
    % predicted egde and ori
    opt.curr_imgId = imgids{idx};
    load([res_mat_dir opt.curr_imgId '_lab_v_g.mat'], 'edge_ori');
    
    res_edge = edge_ori.edge;
    res_ori = edge_ori.ori;
    
%     % predicted orientation before nms
%     PlotOri(res_ori);
%     if opt.print, export_fig(sprintf('%s/%s_ori_res_1', export_dir, imgids{idx}), '-q101', '-transparent', '-depsc3', '-eps', '-tight');  end;
%     
%     % predicted edge before nms
%     figure, imshow(res_edge), truesize(gcf);
%     if opt.print, export_fig(sprintf('%s/%s_edge_res_1', export_dir, imgids{idx}), '-q101', '-transparent', '-depsc3', '-eps', '-tight');  end;
    
    res_edge = edge_nms(res_edge, 0);
    mask = find(res_edge<0.01);
    res_ori(mask) = 0;
    
    % predicted edge
%     figure, imshow(1-res_edge, 'Border','tight'), truesize(gcf);
%     if opt.print, export_fig(sprintf('%s/%s_edge_res', export_dir, imgids{idx}), '-q101', '-transparent', '-depsc3', '-eps', '-tight');  end;
    
    % predicted orientation
%     PlotOri(res_ori);
%     if opt.print, export_fig(sprintf('%s/%s_ori_res', export_dir, imgids{idx}), '-q101', '-transparent', '-depsc3', '-eps', '-tight');  end;
    
    % predicted TP, FP, FN, with occlusion arraw
    res_img = zeros([size(edge_ori.edge), 2], 'single');
    res_img(:,:,1) = edge_ori.edge;
    res_img(:,:,2) = edge_ori.ori; 
    PlotOcclusion(I_white, res_img, gt_img, opt);
    if opt.print, 
        saveas(gcf, sprintf('%s/%s_occ_res_%s.png', export_dir, imgids{idx}, exp_name), 'png');
        close(gcf);
    end;
    
    disp(strcat('plot img ', num2str(opt.curr_imgId), ' finished!'));
    
end
