function rsquare = calcR2(est_y, y)

SSE = mean((est_y-y).^2, 2);
TSS = var(y, [], 2);
rsquare = max(0, 1 - SSE./TSS);
end