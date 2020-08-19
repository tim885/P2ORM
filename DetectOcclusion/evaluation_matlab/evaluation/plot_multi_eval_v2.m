function plot_multi_eval_v2(evalDirs, opt, nms, cols, lines)
% plot precision-recall curve 
% opt = struct();
% opt = CatVarargin(opt, varargin); 
if(nargin<3||isempty(nms)), nms={}; end; if(~iscell(nms)), nms={nms}; end
if(nargin<4||isempty(cols)), cols=repmat({'r','g','b','k','m', 'y', 'c'},1,100); end
if(nargin<5||isempty(lines)), lines=repmat({'-'},1,100); end
if(~iscell(evalDirs)), evalDirs={evalDirs}; end; if(~iscell(cols)), cols={cols}; end

figure();
%%%%%%% plot setting %%%%%%%%%%%%%%%%%%%
box on; grid on; hold on;
line([0 1],[.5 .5],'LineWidth',1.6,'Color',.75*[1 1 1]);
for f=0.1:0.1:0.9, r=f:0.01:1; p=f.*r./(2.*r-f); %f=2./(1./p+1./r)
  plot(r,p,'Color',[0 .8 0]); plot(p,r,'Color',[0 .8 0]); end
set(gca,'XTick',0:0.1:1,'YTick',0:0.1:1,'GridLineStyle',':','GridAlpha',0.6);
xlabel('Recall'); ylabel('Precision');
axis equal; axis([0 1 0 1]);



%%%%%%%% load data %%%%%%%%%%%%%%%%%%%%%%%
n = length(evalDirs); hs=zeros(1,n); res=zeros(n, 8); ars=cell(1,n); % n methods
for i=1:n,
    if exist(fullfile(evalDirs{i}, ['eval_bdry_thr', opt.append, '.txt']), 'file'),
        prvals = dlmread(fullfile(evalDirs{i},['eval_bdry_thr',opt.append,'.txt'])); % thresh, recall, prec., f1
        f = find(prvals(:,2) >= 0.001);
        prvals = prvals(f,:);
        ars{i} = prvals;
        res(i, 1:8) = dlmread(fullfile(evalDirs{i},['eval_bdry',opt.append,'.txt']));
    end
end

[~,o] = sort(res(:,4), 'descend'); res = res(o, :); ars = ars(o); cols = cols(o); 
if ~isempty(nms), nms = nms(o); end
fprintf('%s\n', opt.eval_item_name);

%%%%%%%% plot data %%%%%%%%%%%%%%%%%%%%%
for i=n:-1:1
    % plot smoothed curves 
%     samplingRateIncrease = 10;
%     newXSamplePoints = linspace(min(ars{i}(:,2)), max(ars{i}(:,2)), length(ars{i}(:,2)) * samplingRateIncrease);
%     smoothedY = spline(ars{i}(:,2), ars{i}(:,3), newXSamplePoints);
%     hs(i) = plot(newXSamplePoints, smoothedY, 'Color', cols{i}, 'LineWidth', 2, 'LineStyle', lines{i});
    
    % plot directly 
    hs(i) = plot(ars{i}(:,2), ars{i}(:,3), 'Color', cols{i}, 'LineWidth', 2, 'LineStyle', lines{i});
    
    fprintf('prob_thresh=%.3f ODS=%.3f OIS=%.3f AP=%.3f',res(i,[1 4 7 8]));
%     fprintf('ODS: F( %1.3f, %1.3f ) = %1.3f   [th = %1.3f]\n',res(i, [2:4 1]));
%     fprintf('OIS: F( %1.3f, %1.3f ) = %1.3f\n',res(i, [5:7]));
%     fprintf('AP = %1.3f\n\n',res(i,8));
    if (~isempty(nms)), fprintf(' - %s', nms{i}); end; fprintf('\n');
end

if (isempty(nms)), return; end

for i=1:n
    legend_nms{i} = sprintf('[F=.%i] %s', round(res(i, 4)*1000), nms{i});
end
hold off;
legend(hs, legend_nms, 'Location', 'sw', 'FontSize', 14);

end