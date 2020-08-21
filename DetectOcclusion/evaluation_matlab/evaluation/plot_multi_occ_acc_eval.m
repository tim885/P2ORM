function plot_multi_occ_acc_eval(evalDirs, opt, nms, cols, lines)
% plot accuracy-recall curve as in DOC2015
% opt = struct();
% opt = CatVarargin(opt, varargin); 
if(nargin<3||isempty(nms)), nms={}; end; if(~iscell(nms)), nms={nms}; end
if(nargin<4||isempty(cols)), cols=repmat({'r','g','b','k','m', 'y', 'c'},1,100); end
if(nargin<5||isempty(lines)), lines=repmat({'-'},1,100); end
if(~iscell(evalDirs)), evalDirs={evalDirs}; end; if(~iscell(cols)), cols={cols}; end


figure();
box on; grid on;
set(gca,'XTick',0:0.1:1)
xlabel('Recall '); ylabel('Accuracy');
axis square; axis([0 1 0.5 1]);


n = length(evalDirs); hs=zeros(1,n); res=zeros(n, 8); ars=cell(1,n);
for i=1:n,
    if exist(fullfile(evalDirs{i}, ['eval_bdry_thr', opt.append, '.txt']), 'file'),
        prvals = dlmread(fullfile(evalDirs{i},['eval_bdry_thr',opt.append,'.txt'])); % thresh,r,p,f
        f = find(prvals(:,2) >= 0.001);
        prvals = prvals(f,:);
        ars{i} = prvals;
        res(i, 1:8) = dlmread(fullfile(evalDirs{i},['eval_bdry',opt.append,'.txt']));
    end
end

[~,o] = sort(res(:,4), 'descend'); res = res(o, :); ars = ars(o); cols = cols(o); 
if ~isempty(nms), nms = nms(o); end
fprintf('Occlusion accuracy\n');
hold on;
for i=n:-1:1
    hs(i) = plot(ars{i}(:,2), ars{i}(:,3), 'Color', cols{i}, 'LineWidth', 2, 'LineStyle', lines{i});
    fprintf('ODS=%.3f OIS=%.3f AP=%.3f',res(i,[4 7 8]));
%     fprintf('ODS: F( %1.3f, %1.3f ) = %1.3f   [th = %1.3f]\n',res(i, [2:4 1]));
%     fprintf('OIS: F( %1.3f, %1.3f ) = %1.3f\n',res(i, [5:7]));
%     fprintf('AP = %1.3f\n\n',res(i,8));
    if (~isempty(nms)), fprintf(' - %s', nms{i}); end; fprintf('\n');
end
if (isempty(nms)), return; end
for i=1:n
    legend_nms{i} = sprintf('%s', nms{i});
end
legend(hs, legend_nms, 'Location', 'sw', 'FontSize', 12);
hold off;

end