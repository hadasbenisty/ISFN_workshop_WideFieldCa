%% Animal 1
data_file = '../data/data_animal1.mat';
anatomic_map_file = '../data/atlas.mat';

% load and plot data
load(data_file, 'x', 't', 'behavior_traces', 'behavior_labels', 'func_map');
% taking a shorter part of the experiment to reduce computation time
start_ind = 2251;
end_ind = 6250;
x = x(:, start_ind:end_ind);
t = t(start_ind:end_ind);
behavior_traces = behavior_traces(:, start_ind:end_ind);
fsample = 10; % sampling frequency 
%% Task A: visualize data
% 1. plot the anatomic parcellation map saved in '../data/atlas.mat'
load(anatomic_map_file, 'anatomic_parcels');
B = size(behavior_traces, 1);
figure;
subplot(B+2,2,1);
imagesc(anatomic_parcels);title('Anatomic Parcellation');
% 2. plot the functional parcellation map stored as func_map

% 3. choose the functional parcel corresponding to anatomical visual  
%    cortex (anatomical #1) and corresponding to the motor cortex (anatomical #23) 
%    and plot their traces vs time
v1_parcel = 14;
m1_parcel = 38;
subplot(B+2,2,2);imagesc(func_map);title('Functional Parcellation');
subplot(B+2, 1, 2);plot(t, x([v1_parcel m1_parcel], :)); axis tight;
legend('v1','m1');
% 4. plot behavior traces vs. time
for i = 1:B
subplot(B+2, 1, i+2);plot(t, behavior_traces(i, :));ylabel(behavior_labels{i});
axis tight;
end
xlabel('Time [sec]');


%% Task B: modeling behavior using neuronal activity
Kfolds = 10; % folds for cross validation
% 1. Use the function 'train_test_regression' to train a linear regression
%    for each behavior trace based on neuronal activity using 10-fold
%    cross-validation. What is the R2? Is it "train R2"? or "Test R2"? What 
%    is the difference? 
[r2_x, est_x] = train_test_regression(Kfolds, x, behavior_traces);
% 2. Plot each behavior trace along with the model's prediction (use hold
%    all to plot both on the same axis
figure;
for i=1:B
subplot(B, 1, i);plot(t, zscore(behavior_traces(i, :)), t, est_x(i,:));axis tight;
title([ behavior_labels{i} ' R^2=' num2str(r2_x(i))]);
end
suptitle('Modeling by x(t)');
% 3. Use the function diffmap_euc to reduce the dimension of activity based on
%    diffusion map. Use knn-param = 100. 
% reduce dim - x(t)
[phi_x, Lambda_x1] = diffmap_euc(x, 100);
% 4. Use the function plot_latent to plot the first 3 components of the 
%    Embedded activity as a scatter plot. Colors should be z-scored pupil
%    values.
figure;plot_latent(phi_x, zscore(behavior_traces(1,:)));title('\phi_x');
% 5. Use different values of knn-param and scatter plot the components 
[phi_x2, Lambda_x2] = diffmap_euc(x, 1000);
[phi_x3, Lambda_x3] = diffmap_euc(x, 10);
figure;plot_latent(phi_x2, zscore(behavior_traces(1,:)));title('\phi_x');
% 6. Plot the eigenvalues you got for different values of knn-param 
figure;plot([Lambda_x1 Lambda_x2 Lambda_x3])
% 7. Train a regression model for behavior based on the embedded activity
% using the first 10 components
[r2_phi_x, est_phi_x] = train_test_regression(Kfolds, phi_x(1:10, :), behavior_traces);
% 8. Plot behavior traces and estimated traces vs. time
figure;
subplot(2, 1, 1); plot(t, zscore(behavior_traces(1, :)), t, est_phi_x(1,:))
axis tight;legend('pupil','estimated pupil');
subplot(2, 1, 2); plot(t, zscore(behavior_traces(2, :)), t, est_phi_x(2,:))
axis tight;legend('wheel','estimated wheel');
xlabel('Time [sec]');suptitle('Modeling by \Phi_x');

%% Task C: modeling behavior using neuronal connectivity (Euc. Distance)
% 1. Write a function that extracts Pearson's correlation (use 'corr') using 
%    a sliding window. Input - x, window size (samples) and window hop (samples)
%    Outputs - tC     - connectivity values of parcels over parcels over time points. 
%              t_win  - indices of centers of analysis windows
%    Use this function to extract correlations using 3sec window size and 
%    0.1 window hop 
winsz = 3*fsample; % 3sec -> to samples
winhop = 1; % samples
[tC, t_win] = dynamic_corr(x, winsz, winhop);
% 2. Use the function cmat2feat to reshape tC into a pairwise correlation
%    over time matrix
tC_vec = cmat2feat(tC);
% 3. Reduce the dim of c(t) using diff map (as you did for x(t))
[phi_c_euc, Lambda_c_euc] = diffmap_euc(tC_vec, 100);
% 4. resampling behavior to match windowed c(t)
behavior_traces_c = interp1(t, behavior_traces', t(t_win))';

% 5. Train a regression model for behavior based on the embedded
% correlations using the first 10 components
[r2_phi_c_euc, est_phi_c_euc] = train_test_regression(Kfolds, phi_c_euc(1:10, :), behavior_traces_c);
% 6. Use the function plot_latent to plot the first 3 components of the
% embedded correlations as a scatter where colors are zscored pupil values
figure;plot_latent(phi_c_euc, zscore(behavior_traces_c(1,:)));title('\phi_c (Euc)');
% 7. Plot behavior traces and estimated traces vs. time
figure;
subplot(2, 1, 1); plot(t, zscore(behavior_traces(1, :)), t(t_win), phi_c_euc(1,:))
axis tight;legend('pupil','estimated pupil');
subplot(2, 1, 2); plot(t, zscore(behavior_traces(2, :)), t(t_win), phi_c_euc(2,:))
axis tight;legend('wheel','estimated wheel');
xlabel('Time [sec]');
suptitle('Modeling by \Phi_c (Euc)');
%% Task D: modeling behavior using neuronal connectivity (R-Distance)

% load pre-calculated R-mean
load('../data/Rmean_animal1');
% 1. Use the function getDiffMap_corr to reduce the dimension of
% correlations. To save some time, use the pre-calculated 'mRiemannianMean'
% stored in '../data/Rmean_animal1' as input. Use knn-param=100.
[phi_c_R, proj_c_R, Lambda_c_R] = getDiffMap_corr(tC, 100, mRiemannianMean);
% 2. Model (resampled) behavior using the new embedded correlations
[r2_phi_c_R, est_phi_c_R] = train_test_regression(Kfolds, phi_c_R(1:10, :), behavior_traces_c);
% 3. plot latent dynamics of the first 3 components with colors by pupil
figure;plot_latent(phi_c_R, zscore(behavior_traces_c(1,:)));title('\phi_c (R)');
% 4. Plot behavior traces and estimated traces vs. time
figure;
subplot(2, 1, 1); plot(t, zscore(behavior_traces(1, :)), t(t_win), est_phi_c_R(1,:))
axis tight;legend('pupil','estimated pupil');
subplot(2, 1, 2); plot(t, zscore(behavior_traces(2, :)), t(t_win), est_phi_c_R(2,:))
axis tight;legend('wheel','estimated wheel');
xlabel('Time [sec]');
suptitle('Modeling by \Phi_c (R)');

%% Task E
% Visualize overall modeling results
R2vals = [r2_x  r2_phi_x  r2_phi_c_euc  r2_phi_c_R];
figure;
bar(R2vals);
set(gca, 'XTickLabels', {'pupil', 'wheel'});
legend({'x', '\phi_x', '\phi_c (euc)',   '\phi_c (R)'});
ylabel('R^2');ylim([0 0.5]);

%% Task F - Animal 2
% Clear resuls and repeat for animal 2
clear;
close all;
data_file = '../data/data_animal2.mat';
anatomic_map_file = '../data/atlas.mat';

% load and plot data
load(data_file, 'x', 't', 'behavior_traces', 'behavior_labels', 'func_map');
% taking a shorter part of the experiment to reduce computation time
start_ind = 5001;
end_ind = 8e3;
x = x(:, start_ind:end_ind);
t = t(start_ind:end_ind);
behavior_traces = behavior_traces(:, start_ind:end_ind);
fsample = 10; % sampling frequency 
B = size(behavior_traces, 1);
% visualize data
load(anatomic_map_file, 'anatomic_parcels');
v1_parcel = 8;
m1_parcel = 4;
figure;
subplot(B+2,2,1);
imagesc(anatomic_parcels);title('Anatomic Parcellation');
subplot(B+2,2,2);imagesc(func_map);title('Functional Parcellation');
subplot(B+2, 1, 2);plot(t, x([v1_parcel m1_parcel], :)); axis tight;
legend('v1','m1');
for i = 1:B
subplot(B+2, 1, i+2);plot(t, behavior_traces(i, :));ylabel(behavior_labels{i});
axis tight;
end
xlabel('Time [sec]');

% modeling behavior using x(t)
Kfolds = 10; % folds for cross validation
[r2_x, est_x] = train_test_regression(Kfolds, x, behavior_traces);
figure;
for i=1:B
subplot(B, 1, i);plot(t, zscore(behavior_traces(i, :)), t, est_x(i,:));axis tight;
title([ behavior_labels{i} ' R^2=' num2str(r2_x(i))]);
end
suptitle('Modeling by x(t)');

% extract c(t) using a sliding window
winsz = 3*fsample; % 3sec -> to samples
winhop = 1; % samples
[tC, t_win] = dynamic_corr(x, winsz, winhop);

% reduce dim - x(t)
[phi_x, Lambda_x] = diffmap_euc(x, 100);

% reduce dim - c(t) using Euc. distnace 
tC_vec = cmat2feat(tC);

[phi_c_euc, Lambda_c_euc] = diffmap_euc(tC_vec, 100);
% load pre-calculated R-mean
load('../data/Rmean_animal2');
[phi_c_R, proj_c_R, Lambda_c_R] = getDiffMap_corr(tC, 100, mRiemannianMean);
% resampling behavior to match windowed c(t)
behavior_traces_c = interp1(t, behavior_traces', t(t_win))';

% plot latent dynamics, colors by behavior
figure;
subplot(2,3,1);plot_latent(phi_x, zscore(behavior_traces(1,:)));title('\phi_x');
subplot(2,3,2);plot_latent(phi_c_euc, zscore(behavior_traces_c(1,:)));title('\phi_c (Euc)');
subplot(2,3,3);plot_latent(phi_c_R, zscore(behavior_traces_c(1,:)));title('\phi_c (R)');
subplot(2,3,4);plot_latent(phi_x, zscore(behavior_traces(2,:)));title('\phi_x');
subplot(2,3,5);plot_latent(phi_c_euc, zscore(behavior_traces_c(2,:)));title('\phi_c (Euc)');
subplot(2,3,6);plot_latent(phi_c_R, zscore(behavior_traces_c(2,:)));title('\phi_c (R)');
suptitle('Latent Dynamics, Colors by Pupil - Top, Wheel - Bottom');

% model behavior 
[r2_phi_x, est_phi_x] = train_test_regression(Kfolds, phi_x(1:80, :), behavior_traces);
[r2_phi_c_euc, est_phi_c_euc] = train_test_regression(Kfolds, phi_c_euc(1:10, :), behavior_traces_c);
[r2_phi_c_R, est_phi_c_R] = train_test_regression(Kfolds, phi_c_R(1:10, :), behavior_traces_c);

% visualize modeling results
R2vals = [r2_x  r2_phi_x  r2_phi_c_euc  r2_phi_c_R];
for i = 1:B
figure;
subplot(4, 1, 1);
bar(R2vals(i, :));
set(gca, 'XTickLabels', {'x', '\phi_x', '\phi_c (euc)',   '\phi_c (R)'});
ylabel('R^2');ylim([0 1]);
subplot(4, 1, 2); plot(t, zscore(behavior_traces(i, :)), t, est_phi_x(1,:));
axis tight;ylabel('\phi_x');
subplot(4, 1, 3); plot(t, zscore(behavior_traces(i, :)), t(t_win), est_phi_c_euc(1,:));
axis tight;ylabel('\phi_c (Euc)');
subplot(4, 1, 4); plot(t, zscore(behavior_traces(i, :)), t(t_win), est_phi_c_R(1,:))
axis tight;ylabel('\phi_c (R)');
suptitle(['Modeling ' behavior_labels{i}]);
end


