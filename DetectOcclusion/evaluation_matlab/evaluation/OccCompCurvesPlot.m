% This script plots multi algorithms boundary PR curves, 
% object occlusion boundary PR curves, occlusion boundary AR curves.
% only BSDSownership dataset is supported. 
addpath(genpath('../export_fig'));
clear;

DataSet = 'BSDSownership';
method_folder = '2020-01-15_15-40-52/results_vis';
export_dir = fullfile(pwd, '..', '..', 'experiments', 'output', method_folder, 'test_19_BSDSownership/res_mat/test_pred/eval_curves');
        
algs = {fullfile(pwd, '..', '..', 'experiments', 'output', method_folder, 'test_19_BSDSownership/res_mat/test_pred'),...
        fullfile(pwd, '..', '..', 'experiments', 'output', 'other_methods_results', 'BSDS_ofnet_offered/res_mat/test_pred'),...    
        fullfile(pwd, '..', '..', 'experiments', 'output', 'other_methods_results', 'BSDS_doobnet_offered/res_mat/test_pred'),...
        fullfile(pwd, '..', '..', 'experiments', 'output', 'other_methods_results', 'BSDS_doc-hed_offered/res_mat/test_pred'),...
        fullfile(pwd, '..', '..', 'experiments', 'output', 'other_methods_results', 'BSDS_doc-dmlfov_offered/res_mat/test_pred'),...
        fullfile(pwd, '..', '..', 'experiments', 'output', 'other_methods_results', 'BSDS_srf-occ_offered/res_mat/test_pred')};

nms = {'Ours',...
       'OFNet',...
       'DOOBNet',...
       'DOC-HED',...
       'DOC-DMLFOV',...
       'SRF-OCC'};

close all;

app_name = '_occ'; 
opt.print = 1;
set_line_color_style;

opt.eval_item_name = 'Boundary';  % occlusion boundary Precision-Recall curve
opt.append = [app_name, '_e'];  plot_multi_eval_v2(algs, opt, nms, colors, lines); 
set(gcf, 'units', 'inches', 'position', [5 5 6 6])
if opt.print, export_fig(sprintf('%s%s_edge_pr', export_dir, DataSet), '-q101', '-transparent', '-depsc3', '-eps', '-tight');  end;

opt.eval_item_name = 'Orientation PR';  % oriented occlusion boundary Precision-Recall curve
opt.append = [app_name, '_poc'];  plot_multi_eval_v2(algs, opt, nms, colors, lines); 
set(gcf, 'units', 'inches', 'position', [5 5 7.5 7.5])
if opt.print, export_fig(sprintf('%s%s_edge_ori_pr', export_dir, DataSet), '-q101', '-transparent', '-depsc3', '-eps', '-tight');  end;

opt.append = [app_name, '_aoc'];   % oriented occlusion boundary Accuracy-Recall curve
plot_multi_occ_acc_eval(algs, opt, nms, colors, lines); 
set(gcf, 'units', 'inches', 'position', [5 5 6 6])
if opt.print, export_fig(sprintf('%s%s_occ_aor', export_dir, DataSet), '-q101', '-transparent', '-depsc3', '-eps', '-tight');  end;

disp('finish ploting!')

