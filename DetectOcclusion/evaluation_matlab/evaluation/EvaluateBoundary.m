function EvaluateBoundary(imglist, varargin)
opt = struct('Evaluate', 'edge', 'resPath', 'gtPath', 'nthresh', 30, ...,
    'renormalize', 1, 'w_occ', 0, 'overwrite', 0, 'dataBase', 'PASCAL', 'scale_id', 0);

opt = CatVarargin(opt, varargin);
append = opt.append;

if exist(fullfile(opt.outDir,['eval_bdry_thr',append,'.txt']), 'file') & ~ opt.overwrite
    return;
end

outDir = opt.outDir;
resPath = opt.resPath;
gtPath = opt.gtPath;
opt.dilate = 0; % do not dilate the ground truth for evaluation 
scale_id = opt.scale_id; 
if scale_id ==0; scale_id = 1; end 

if opt.w_occ
    fname = fullfile(opt.outDir, ['eval', append, '_acc.txt']); 
else  % only boundary 
    fname = fullfile(opt.outDir, ['eval_bdry',append,'.txt']); 
end

if ~exist(fname, 'file')  | opt.overwrite 
    % assert(length(resfile) == length(gtfile));
    n = length(imglist);
    do = false(1, n);  %  list to do  
    params = cell(1, n);
    
    for ires = 1:length(imglist)
        prFile = fullfile(outDir, [imglist{ires}, '_ev1',append,'.txt']);  % out eval file 
        if exist(prFile, 'file')& ~opt.overwrite; continue; end 
        do(ires) = true;
        
        % get results
        if strcmp(opt.method_folder, 'DOOBNet')
            resfile = fullfile(resPath, [imglist{ires}, '.mat']);
        else  
            resfile = fullfile(resPath, [imglist{ires}, '_lab_v_g.mat']);  % by xuchong
        end
        
        edge_maps = load(resfile);
        edge_maps = edge_maps.edge_ori;
        
        if opt.w_occ  
            % w/ occ orientation
            res_img = zeros([size(edge_maps.edge), 2], 'single');
            res_img(:,:,1) = edge_maps.edge;  % [0,1]
            res_img(:,:,2) = edge_maps.ori;  % [-pi,pi] in rad
        else  
            % only edge
            res_img = zeros([size(edge_maps.edge), 1], 'single');
            res_img(:,:,1) = edge_maps.edge;
        end
        
        if size(edge_maps,2) >= 3 && opt.w_occ  % perhaps scale?
            opt.rank_score = res_img(:,:,1) + edge_maps{scale_id,3};
        end
        
        % get ground truth
        switch opt.DataSet
            case 'PIOD'
                gtfile = fullfile(gtPath, [imglist{ires}, '.mat']);
                bndinfo_pascal = load(gtfile);
                bndinfo_pascal = bndinfo_pascal.bndinfo_pascal;         
                gt_img = GetEdgeGT(bndinfo_pascal);

            case 'BSDSownership'
                gtfile = fullfile(gtPath, [imglist{ires}, '.mat']);
                
                if opt.own_gt  
                    % by xuchong, use gt in exp res dir 
                    gtStruct = load(gtfile);
                    gt_img = gtStruct.gt_edge_theta;  % edge+ori; H,W,(2+2)
                else
                    gtStruct = load(gtfile);
                    gtStruct = gtStruct.gtStruct;
                    gt_img = cat(3, gtStruct.gt_theta{:});  % edge+ori; H,W,(2+2)
                end       
                
            case 'ibims'
                gtfile = fullfile(gtPath, [imglist{ires}, '.mat']);
                gtStruct = load(gtfile);
                gt_img = gtStruct.gt_edge_theta;  % edge+ori; H,W,(2+2)
                
            case 'nyuv2'
                gtfile = fullfile(gtPath, [imglist{ires}, '.mat']);
                gtStruct = load(gtfile);
                gt_img = gtStruct.gt_edge_theta;  % edge+ori; H,W,(2+2)
        end
        
%         if opt.vis 
%             imagesc(gt_img(:,:,2)); 
%             pause; 
%         end
        
        if ~all(size(res_img(:,:,1)) - size(gt_img(:,:,1)) == 0)  % resize to gt size
            res_img = imresize(res_img, size(gt_img(:,:,1)), 'bilinear'); 
            opt.rank_score = imresize(opt.rank_score, size(gt_img(:,:,1)), 'bilinear'); 
        end
        params{ires} = {res_img, gt_img, prFile, opt};
        %EvaluateSingle(res_img, gt_img, prFile, opt);
    end

    params = params(do);
    n = length(params);
    
    % eval for loop
%     for i=1:n,
%         EvaluateSingle(params{i}{:});
%     end
    
    % parallel tool for eval
    parforProgress(n);  
    parfor i=1:n,
        feval('EvaluateSingle', params{i}{:});
        parforProgress;
    end
    parforProgress(0);
end

%% collect results from single image evaluation
if strfind(append, '_occ');  % eval edge, OPR, AOR
    % opt.overwrite = 1;
    collect_eval_bdry_occ(opt.outDir, append, opt.overwrite);
else  % eval only edge 
    collect_eval_bdry_v2(opt.outDir, append);
end

%% clean up
% system(sprintf('rm -f %s/*_ev1%s.txt',opt.outDir,append));

end



