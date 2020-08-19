function [m1, m2] = correspondOccPixels(match1, occ, gt_occ) 
    % match1 is the matched pixel of bmap1 indexed with corresponding pixel
    % of bmap2 
    m1 = false(size(match1)); 
    m2 = false(size(match1)); 
    bmap = find(match1(:)>0);   
    gt_ind = match1(bmap);   
    
    theta = occ(bmap);  % occ ori 
    gt_theta = gt_occ(gt_ind); 
    ind = match_theta(theta, gt_theta); 
    m1(bmap(ind)) = 1; 
    m2(gt_ind(ind)) = 1; 

end

function ind = match_theta(theta, gt_theta)
% find pixel ind satisfying occ ori metric
% theta, gt_theta in rad

abs_diff = abs(theta-gt_theta); 
abs_diff = mod(abs_diff, 2*pi); 
ind = find(abs_diff  <= pi/2 |  (abs_diff  >= 3*pi/2)); 

end