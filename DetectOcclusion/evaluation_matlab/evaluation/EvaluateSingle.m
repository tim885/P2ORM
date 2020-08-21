function [thresh,cntR,sumR,cntP,sumP, VB, VO] = EvaluateSingle(pb, gt, prFile, varargin)
% [thresh,cntR,sumR,cntP,sumP, VB, VO] = EvaluateSingle(pb, gt, prFile, varargin)
%
% Calculate precision/recall curve.
%
% INPUT
%	pb  :       [N,M,2], prediction results
%               the first is soft boundary map
%               the second is occlusion orientation map
%
%	gt	:       Corresponding groundtruth
%               [N,M,2*t], where t is the number of gt annotation
%
%   prFile  :   Temporary output for this image.
%
% OUTPUT
%	thresh		Vector of threshold values.
%	cntR,sumR	Ratio gives recall.
%	cntP,sumP	Ratio gives precision.
%   VB, VO      Visulization for Bounadry and Occlusion orientation results.

opt = struct('nthresh', 99, 'maxDist', 0.0075, 'thinpb', 1, ...,
    'w_occ', 0, 'renormalize', 1, 'rank_score', [], 'debug', 0, 'fastmode', 0); 
opt = CatVarargin(opt, varargin); 
nthresh = opt.nthresh; 
maxDist = opt.maxDist; 
thinpb = opt.thinpb; 
rank_score = opt.rank_score; 

thresh = linspace(1/(nthresh+1),1-1/(nthresh+1),nthresh)';
% zero all counts
if opt.w_occ
cntR_occ = zeros(size(thresh));
cntP_occ = zeros(size(thresh));
end
cntR = zeros(size(thresh));

cntP = zeros(size(thresh));
sumP = zeros(size(thresh));

pb_num = size(pb,3) ; 
if pb_num >= 2 && opt.w_occ
    occ = pb(:,:,2);  % occ ori 
    pb = pb(:,:,1);  % occ edge prob 
end

if size(gt,3) >= 2 || opt.w_occ
    gt_occ = gt(:,:,2:2:end); % gt occ ori
    gt = gt(:,:,1:2:end);  % gt edge
end

if thinpb
        % perform nms operation
        pb = edge_nms(pb, 0);
        if opt.debug; imshow(1-pb); pause; end 
        
        % save prob after nms
        img_dir = strrep(opt.resPath, 'res_mat/test_pred', 'images');
        prFile_cells = strsplit(prFile, '/'); 
        file_name_cells = strsplit(prFile_cells{end}, '_');
        cell_sz = size(file_name_cells);
        % disp(cell_sz(2)); 
        if cell_sz(2) == 4
            % ibims sample name 
            sample_name = strcat(file_name_cells{1}, '_', file_name_cells{2}, '-rgb');
        else
            sample_name = file_name_cells{1};
        end

        out_f = fullfile(img_dir, strcat(sample_name, '_lab_v_g_nms.png'));
        
        pb_img = cast(pb * 255., 'uint8');
        imwrite(pb_img, out_f);
end

if ~isempty(rank_score)
    pb(pb > 0) = rank_score(pb>0);   
end 
    
if opt.renormalize
    id = pb > 0;  
    pb(id) = (pb(id) - min(pb(id)))/(max(pb(id))-min(pb(id))+eps); 
end

sumR = 0;
if(nargout>=6), VB=zeros([size(pb) 3 nthresh]); end
if(nargout>=7), VO=zeros([size(pb) 4 nthresh]); end
for i = 1:size(gt,3),
  sumR = sumR + sum(sum(gt(:,:,i)));
end
sumR = sumR .* ones(size(thresh));
% match1_all = zeros(size(gt)); 

% Note that: The fastmode written by Peng Wang, which
% refer to https://github.com/pengwangucla/MatLib/blob/master/Evaluation/EvaluateBoundary/EvaluateSingle.m#L82
% And the result will drop. You can only use to test.
% The other statistical method modified by Guoxia Wang from 
% https://github.com/pdollar/edges/blob/master/edgesEvalImg.m#L61
if (opt.fastmode)  
    for t = nthresh:-1:1, 
        % pb(pb<thresh(t)) = 0; 
        bmap = pb >= thresh(t); 
        if t<nthresh,
            % consider only new boundaries
            bmap = bmap .* ~(pb>=thresh(t+1));
            % these stats accumulate
            cntR(t) = cntR(t+1);
            cntP(t) = cntP(t+1);
            sumP(t) = sumP(t+1);

            if opt.w_occ
                cntR_occ(t) = cntR_occ(t+1); 
                cntP_occ(t) = cntP_occ(t+1); 
            end
        end

        % bmap = RmSmallEdge(bmap, 5);
        % accumulate machine matches, since the machine pixels are
        % allowed to match with any segmentation 
        accP = zeros(size(bmap));
        if opt.w_occ;  accP_occ = accP; end 
        % compare to each seg in turn 
        for i = 1:size(gt,3),
            % compute the correspondence
            [match1, match2] = correspondPixels(double(bmap), double(gt(:,:,i)), maxDist);

            if opt.w_occ
                % match1_all(:,:,i) = match1_all(:,:,i)+match1;
                % reserver the edge that also correct with directions 
                [match1_occ,match2_occ] = correspondOccPixels(match1, ...,
                    occ, gt_occ(:,:,i));
            end

            gt(:,:,i) = gt(:,:,i).*~match2; 
           % compute recall
            cntR(t) = cntR(t) + sum(match2(:)>0); 
            accP = accP | match1; 

            if opt.w_occ 
                gt_occ(:,:,i) = gt_occ(:,:,i).*~match2; 
                cntR_occ(t) = cntR_occ(t) + sum(match2_occ(:)>0);
                accP_occ = accP_occ | match1_occ; 
            end
        end

        if opt.w_occ
            cntP_occ(t) = cntP_occ(t) + sum(accP_occ(:));
        end

        % compute precision
        sumP(t) = sumP(t) + sum(bmap(:));
        cntP(t) = cntP(t) + sum(accP(:));
    end
else  % standard mode
    for t = 1:nthresh, 
        bmap = double(pb >= max(eps,thresh(t)));  % binary edge map satisfying thresh  
        Z = zeros(size(pb)); 
        matchE = Z;  % correctly labeled boundary matrix 
        matchG = Z; 
        allG = Z;  % H,W
        if opt.w_occ;  
            matchO  = Z;  % correctly labeled occlusion orientation  
            matchGO = Z; 
        end 
        
        n = size(gt,3);  % # of annotators 
        for i = 1:n,
            [match1, match2] = correspondPixels(bmap, double(gt(:,:,i)), maxDist);
            
            if opt.w_occ  % w/ occ ori
                [match1_occ,match2_occ] = correspondOccPixels(match1, ...,
                    occ, gt_occ(:,:,i));
            end

            if opt.w_occ 
                matchO  = matchO | match1_occ > 0; 
                matchGO = matchGO + double(match2_occ>0);
            end

            matchE = matchE | match1>0;
            matchG = matchG + double(match2>0);
            allG = allG + gt(:,:,i);
        end
        
        if opt.w_occ
            cntP_occ(t) = nnz(matchO(:));
            cntR_occ(t) = sum(matchGO(:));
        end
        sumP(t) = nnz(bmap);  % all boundary detection pixel num    
        cntP(t) = nnz(matchE);  % correctly labeled boundary pixel num
        cntR(t) = sum(matchG(:));

        % optinally create visualization of matches
        if(nargout<6), continue; end; 
        cs = [1 0 0; 0 .7 0; .7 .8 1]; cs = cs-1;
        FP = bmap - matchE; TP = matchE; FN = (allG - matchG) / n;
        for g=1:3, VB(:,:,g,t)=max(0,1+FN*cs(1,g)+TP*cs(2,g)+FP*cs(3,g)); end
        VB(:,2:end,:,t) = min(VB(:,2:end,:,t),VB(:,1:end-1,:,t));
        VB(2:end,:,:,t) = min(VB(2:end,:,:,t),VB(1:end-1,:,:,t));

        if (nargout<7), continue; end;
        % FN1 are false negative boundaries pixels
        % FN2 are correctly labeled boundaries but incorrect occlusion
        % orientation
        FP = bmap - matchO; 
        TP = matchO;
        FN1 = (allG - matchG)/n - (matchE - matchO);
        FN2 = matchE - matchO;
        VO(:,:,1,t)=TP; VO(:,:,2,t)=FP; VO(:,:,3,t)=FN1; VO(:,:,4,t)=FN2;
    end
end
% output
fid = fopen(prFile,'w');
if fid==-1,
    error('Could not open file %s for writing.', prFile);
end
if opt.w_occ
    fprintf(fid,'%10g %10g %10g %10g %10g %10g %10g\n',[thresh, cntR, sumR, cntP, sumP, cntR_occ, cntP_occ]');
else
    fprintf(fid,'%10g %10g %10g %10g %10g\n',[thresh cntR sumR cntP sumP]');
end
    
fclose(fid);
end 

