function [phi, Lambda_x] = diffmap_euc(x, knnparam)


MM = min(1000, size(x, 2));
idnsrand = randperm(size(x, 2));
idnsrand = idnsrand(1:MM);
x_p = x(:, idnsrand);
[aff_mat, sig] = calc_affinity_euc(x_p', knnparam);
dParams.maxInd = min(size(aff_mat,1),101);
[~, Lambda_x, Psi_p] = calcDiffusionMap(aff_mat,dParams);
Psi_tr = nystrom_extension(Psi_p, x_p, x, sig, Lambda_x)';
phi=Psi_tr(:,2:end).';