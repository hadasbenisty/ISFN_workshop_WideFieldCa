function [tC, t_win] = dynamic_corr(x, winsz, winhop)




winst = 1:winhop:size(x, 2);
winen = winst+winsz-1;
winst=winst(winen<size(x,2));
winen=winen(winen<size(x,2));
t_win = round((winst+winen)/2);
s=svd(x(:,1:winen));
th = quantile(s,0.3);
tau =median(s(s<=th));
Np = size(x,1);
tC = zeros(Np, Np, length(winst));
for i=1:length(winst)
    W = corr(x(:, winst(i):winen(i))')+eye(Np)*tau;
    tC(:,:,i) = (W+W')/2;
end
end
