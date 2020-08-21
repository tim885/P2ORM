function [VB, VO] = GetThreshSamples(pb, gt, varargin)

opt = struct('thresh', 0.5, 'maxDist', 0.0075, 'thinpb', 1, ...,
    'w_occ', 0, 'rank_score', [], 'debug', 0); 
opt = CatVarargin(opt, varargin); 
thresh = opt.thresh; 
maxDist = opt.maxDist; 
thinpb = opt.thinpb; 
rank_score = opt.rank_score; 

% zero all counts

pb_num = size(pb,3) ; 
if pb_num >= 2 && opt.w_occ
    occ = pb(:,:,2); 
    pb = pb(:,:,1); 
end

if size(gt,3) >= 2 && opt.w_occ
    % first is gt edge, second is gt occ 
    gt_occ = gt(:,:,2:2:end); 
    gt = gt(:,:,1:2:end); 
end

%
if thinpb
        pb=double(edge_nms(pb, 0));        
        if opt.debug; imshow(1-pb); pause; end 
end

if ~isempty(rank_score);
    pb(pb > 0) = rank_score(pb>0);   
end 
    
VB=zeros([size(pb) 3]);
VO=zeros([size(pb) 4]);

bmap = double(pb >= max(eps,thresh));  
Z=zeros(size(bmap)); matchE=Z; matchG=Z; allG=Z;
if opt.w_occ;  accP_occ = Z; matchGO=Z; end 
n = size(gt,3);
for i = n,
    [match1, match2] = correspondPixels(bmap, double(gt(:,:,i)), maxDist);
    if opt.w_occ
        [match1_occ,match2_occ] = correspondOccPixels(match1, ...,
            occ, gt_occ(:,:,i));
    end

    if opt.w_occ 
        accP_occ = accP_occ | match1_occ; 
        matchGO = matchGO + double(match2_occ>0);
    end

    matchE = matchE | match1>0;
    matchG = matchG + double(match2>0);
    allG = allG + gt(:,:,i);
end

% optinally create visualization of matches
cs=[1 0 0; 0 .7 0; .7 .8 1]; cs=cs-1;
FP=bmap-matchE; TP=matchE; FN=(allG-matchG)/n;
for g=1:3, VB(:,:,g)=max(0,1+FN*cs(1,g)+TP*cs(2,g)+FP*cs(3,g)); end
VB(:,2:end,:) = min(VB(:,2:end,:),VB(:,1:end-1,:));
VB(2:end,:,:) = min(VB(2:end,:,:),VB(1:end-1,:,:));

if opt.debug; imshow(VB); pause; end 

if opt.w_occ     
    FP=bmap-accP_occ; TP=accP_occ;
    % FN1 are false negative boundaries
    % FN2 are correctly labeled boundaries but incorrect occlusion
    FN1=(allG-matchG)/n-(matchE-accP_occ);
    FN2=matchE-accP_occ;
    VO(:,:,1)=TP; VO(:,:,2)=FP; VO(:,:,3)=FN1; VO(:,:,4)=FN2;
end
end


