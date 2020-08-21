function collect_eval_bdry_occ(pbDir, append, overwrite)

%overwrite = 1;
fname = fullfile(pbDir, ['eval_bdry',append,'_e.txt']);
if (length(dir(fname))==1) && ~overwrite
    
    return;
    
else
    
    S = dir(fullfile(pbDir,['*_ev1',append,'.txt']));

    % deduce nthresh from .pr files
    for i = 1:length(S)
        filename = fullfile(pbDir,S(i).name);
        AA = dlmread(filename);
        AA_edge = AA(:, 1:5);
        
        AA_ori = AA(:, [1,2,3,7,4]);
        AA_pro = AA(:, [1,2,3,7,5]);
        
        dlmwrite(fullfile(pbDir, [S(i).name(1:end-4), '_e.txt']), AA_edge, ' '); 
        dlmwrite(fullfile(pbDir, [S(i).name(1:end-4), '_aoc.txt']), AA_ori, ' ');
        dlmwrite(fullfile(pbDir, [S(i).name(1:end-4), '_poc.txt']), AA_pro, ' ');

    end
    collect_eval_bdry_v2(pbDir, [append, '_e']);
    collect_eval_bdry_v2(pbDir, [append, '_aoc']);
    collect_eval_bdry_v2(pbDir, [append, '_poc']);
end