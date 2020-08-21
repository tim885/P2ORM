function PlotOri(ori_map)
figure, imagesc(ori_map);
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
axis off;
truesize(gcf);
end