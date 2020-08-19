function sem_map = GetEdgeGT(bndinfo)
% sem_map = GetEdgeGT(bndinfo)
% get boundary and occlusion orientation ground truth map from bndinfo
% struct
if bndinfo.ne == 0
    sem_map = zeros([bndinfo.imsize, 2], 'single');
else
    % generate prediction ground truth map
    sem_map = zeros(bndinfo.imsize, 'single');
    edgemap = zeros(bndinfo.imsize, 'single');
    for iedge = 1:length(bndinfo.edges.indices);
        edgemap(bndinfo.edges.indices{iedge}) = 1;
    end

    sem_map(:,:,1) = edgemap;
    sem_map(:,:,2) = bndinfo.OrientMap;
end

end