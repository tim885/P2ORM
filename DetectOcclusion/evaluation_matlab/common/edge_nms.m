function thin_edge = edge_nms(edge_pr, thresh)
% edge_pr: edge detection prob; [0~1]
% thresh: prob thresh to determine background region

edge_pr = single(edge_pr);
bw_edge = edge_pr <= thresh;
edge_pr(bw_edge) = 0;

[Ox, Oy]  = gradient2(convTri(edge_pr, 4));
[Oxx, ~]  = gradient2(Ox); 
[Oxy,Oyy] = gradient2(Oy);

% cal global orientation
O = mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)), pi);

thin_edge = edgesNmsMex(edge_pr, O, 1, 5, 1.01, 4);
