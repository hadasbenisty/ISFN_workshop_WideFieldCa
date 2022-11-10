function [r2, y_est] = train_test_ridge(Kfolds, X, y, ridge_param)


X = zscore(X')';
y = zscore(y')';
y_est = zeros(size(y));
T = size(y, 2);


chunksize = floor(T/Kfolds);
for chunk_i = 1:Kfolds
    testinds = (1 + (chunk_i - 1)*chunksize):(chunk_i*chunksize);
    traininds = setdiff(1:T, testinds);
    
    y_tr = y(:, traininds);
    X_tr = X(:, traininds);
    X_te=X(:,  testinds);
    
    for i = 1:size(y, 1)
        b = ridge(y_tr(i, :)', X_tr', ridge_param,0);
        y_est(i, testinds) = b(1) + X_te'*b(2:end);
        
    end
end
r2 = calcR2(y_est, y);

end