function plot_latent(phi, y)

scatter(phi(1, :), phi(2, :), 10, zscore(y), 'Filled');
xlabel('\phi_1');
ylabel('\phi_2');
