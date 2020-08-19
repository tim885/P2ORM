% This script is the entry point to perform the evaluation and plot curves 
addpath('../common');
clear;
tic;
curr_path = mfilename('fullpath');

DataSet = 'BSDSownership';
opt.method_folder = '2020-01-15_15-40-52/results_vis';
which_epochs = {19};

%DataSet = 'ibims';
% DataSet = 'nyuv2';

for epoch_id = 1:1:length(which_epochs) 
   
    which_epoch = which_epochs{epoch_id};
    disp('eval epoch: ');
    disp(which_epoch)
    res_rel_dir = fullfile(opt.method_folder, strcat('test_', num2str(which_epoch), '_', DataSet), 'res_mat');
    disp('load result path:')
    disp(res_rel_dir)

    switch DataSet
        case 'BSDSownership'
            oriGtPath  = fullfile(pwd, '..', '..', 'experiments', 'output', res_rel_dir, 'test_gt');
            oriResPath = fullfile(pwd, '..', '..', 'experiments', 'output', res_rel_dir, 'test_pred');
            opt.model_name  = 'BSDSownership';  
            testIdsFilename = fullfile(pwd, '..', '..', '/data/BSDS300/BSDS_theta/test_ori_iids.txt');
            ImageList = textread(testIdsFilename, '%s'); omit_id = []; 

        case 'ibims'
            oriGtPath  = fullfile(pwd, '..', '..', 'experiments', 'output', res_rel_dir, 'test_gt');
            oriResPath = fullfile(pwd, '..', '..', 'experiments', 'output', res_rel_dir, 'test_pred');
            opt.model_name  = 'ibims';  
            testIdsFilename = fullfile(pwd, '..', '..', '/data/ibims/test_ori_iids.txt');
            ImageList = textread(testIdsFilename, '%s'); omit_id = []; 

        case 'nyuv2'
            oriGtPath  = fullfile(pwd, '..', '..', 'experiments', 'output', res_rel_dir, 'test_gt');
            oriResPath = fullfile(pwd, '..', '..', 'experiments', 'output', res_rel_dir, 'test_pred');
            opt.model_name  = 'nyuv2';  
            testIdsFilename = fullfile(pwd, '..', '..', '/data/dataset_real/NYUv2/test_iids.txt');
            ImageList = textread(testIdsFilename, '%s'); omit_id = []; 
    end
    ImageList(omit_id) = [];

    % respath = [oriResPath, opt.method_folder, '/', opt.model_name , '/'];
    respath = oriResPath;
    evalPath = fullfile(oriResPath, opt.method_folder, 'eval_fig'); 
    if ~exist(evalPath, 'dir') 
        mkdir(evalPath); 
    end
    
    % set eval opts
    opt.own_gt = 1;  % eval based on gt in res dir, by xuchong  
    opt.DataSet = DataSet; 
    opt.vis = 1; 
    opt.print = 0; 
    opt.overwrite = 1; 
    opt.visall = 0; 
    opt.append = '';  % eval type: edge, occ_orientation, ...
    opt.validate = 0;  % only val specific num 
    opt.occ_scale = 1;  % set which scale output for occlusion
    opt.w_occ = 1;  % with occ orientation eval 
    if opt.w_occ
        opt.append = '_occ'; 
    end 
    
    opt.scale_id = 0; 
    if opt.scale_id ~= 0
        opt.append = [opt.append, '_', num2str(opt.scale_id)]; 
    end
    
    opt.outDir = respath;
    opt.resPath = respath;
    opt.gtPath = oriGtPath; 
    opt.nthresh = 99;   % threshold to calculate precision and recall
                        % it set to 33 in DOC for save runtime but 99 in DOOBNet.
    opt.thinpb = 1;     % thinpb means performing nms operation before evaluation.
    opt.renormalize = 0; 
    opt.fastmode = 0;   % see EvaluateSingle.m, PrecRecall curve and viz res

    if (~isfield(opt, 'method') || isempty(opt.method)), opt.method = opt.method_folder; end
    fprintf('Starting evaluate %s %s, model: %s and %s\n', DataSet, opt.method, opt.model_name, opt.append); 

    if opt.validate
        valid_num = 10; 
        ImageList = ImageList(1:valid_num); 
    end 

    % eval on dataset and save scores
    EvaluateBoundary(ImageList, opt);

    if opt.vis  % plot res
        close all;
        if strfind(opt.append, '_occ') 
            app_name = opt.append; 
            
            % plot Boundary PR curve  
            opt.eval_item_name = 'Boundary';  
            opt.append = [app_name, '_e'];  
            plot_multi_eval_v2(opt.outDir, opt, opt.method); title('Edge'); 
            
            % plot Occlusion PR curve(OPR curve in paper)
            opt.eval_item_name = 'Orientation PR';   
            opt.append = [app_name, '_poc'];  
            plot_multi_eval_v2(opt.outDir, opt, opt.method);  title('PRO');
            
            % plot occ accuracy w.r.t boundary recall(AOR curve used in DOC2015)
            opt.append = [app_name, '_aoc'];  
            plot_multi_occ_acc_eval(opt.outDir, opt, opt.method);
        else
            % plot Boundary PR curve
            opt.eval_item_name = 'Boundary';  
            plot_multi_eval_v2(opt.outDir, opt, opt.method); title('Edge'); 
        end
    end
    toc;

end
