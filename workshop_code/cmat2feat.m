function tC_vec = cmat2feat(tC)
pairs = nchoosek(1:size(tC,1), 2);
tC_vec = zeros(size(pairs, 1), size(tC,  3));
for p=1:size(pairs,1)
    tC_vec(p, :) = squeeze(tC(pairs(p, 1), pairs(p, 2), :));
end
end