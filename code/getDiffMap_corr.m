function  [phi_c_R, proj_c_R, Lambda_c_R, ...
    mRiemannianMean] = getDiffMap_corr(tC, knnparam, mRiemannianMean)
% dim reduction of x(t) and c(t) using Euc dist. For c(t) also using R-dist
% Input - 
%       x               - neuronal activity, x(t), is #parcels over time
%       tC              - correlation matrices, c(t), #parcels over #parcels over time
%       knnparam        - #of nearest neighbors for bandwidth of diffusion kernel (default - 20) 
%
% Output - 
%       phi_x           - diffusion embedding of x(t)
%       Lambda_a        - spectrum of x(t)
%       phi_c_euc       - diffusion embedding of c(t) using Euc norm
%       Lambda_c_euc    - spectrum of c(t) using Euc norm
%       phi_c           - diffusion embedding of c(t) using R-norm
%       proj_c          - projections of c(t) onto R-cone
%       Lambda_c        - spectrum of c(t) using R-norm
%       mRiemannianMean - R-mean of c(t)
%
%

for i = 1:size(tC, 3)
    tC(:, :, i) = tC(:,:,i)+ eye(size(tC(:,:,i), 1))*0.1;
end
    
%% embedd c(t) using R-norm
MM = min(1000, size(tC, 3));
idnsrand = randperm(size(tC, 3));
idnsrand = idnsrand(1:MM);
if ~exist('mRiemannianMean', 'var')
    mRiemannianMean = RiemannianMean(tC(:,:, idnsrand));
end

proj_c_R = proj_R1(mRiemannianMean^(-1/2),tC);

proj_c_p = proj_c_R(:, idnsrand);
[aff_mat, sig] = calc_affinity_euc(proj_c_p', knnparam);

dParams.maxInd = min(size(aff_mat,1),101);

[~, Lambda_c_R, Psi_p] = calcDiffusionMap(aff_mat,dParams);
Psi_tr = nystrom_extension(Psi_p, proj_c_p, proj_c_R, sig, Lambda_c_R)';
phi_c_R=Psi_tr(:,2:end).';

end



function [M, tC] = RiemannianMean(tC)
Np=size(tC,3);

M  = mean(tC, 3);
for ii = 1 : 200
    A = M ^ (1/2);      %-- A = C^(1/2)
    B = A ^ (-1);       %-- B = C^(-1/2)
    NN=0;
    S = zeros(size(M));
    for jj = 1 : Np
        C = tC(:,:,jj);
        del = logm(B * C * B);
        if any(~isreal(del(:)))
            error('Corr matrix is not invertible');
        end
        S = S + A * del * A;
        NN=NN+1;
    end
    S = S / NN;
    
    M = A * expm(B * S * B) * A;
    
    eps = norm(S, 'fro');
    disp(eps);
    if (eps < 1e-6)
        break;
    end
end
if eps>1e-6
    error('Corr matrix is not invertible');
    
end
end
function mX = proj_R1(mCref,tC)
K  = size(tC, 3);
M  = size(tC,1);
MM = M * (M + 1) / 2;
mX = zeros(MM, K);

mW = sqrt(2) * ones(M) - (sqrt(2) - 1) * eye(M);
for kk = 1 : K
    Skk      = logm(mCref * tC(:,:,kk) * mCref) .* mW;
    if all(~isreal(Skk(:)))
        mX=nan;
        return;
    end
    %
    mX(:,kk) = Skk(triu(true(size(Skk))));
end
end

