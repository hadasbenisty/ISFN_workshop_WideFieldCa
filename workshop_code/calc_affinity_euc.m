function [aff_mat, sigma] = calc_affinity_euc(data, knnparam)
euc_dist = squareform(pdist(data));
nn_dist = sort(euc_dist.').';
knnparam = min(knnparam, size(nn_dist, 2));
sigma = median(reshape(nn_dist(:, 1:knnparam), size(nn_dist,1)*knnparam,1));

aff_mat = exp(-euc_dist.^2/(2*sigma^2));
end