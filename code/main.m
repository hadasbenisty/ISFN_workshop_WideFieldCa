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
%% Task 1: visualize data
% a. plot the anatomic parcellation map saved in '../data/atlas.mat'
load(anatomic_map_file, 'anatomic_parcels');
% b. plot the functional parcellation map stored as func_map

% c. choose the functional parcel corresponding to anatomical visual  
%    cortex (anatomical #1) and corresponding to the motor cortex (anatomical #23) 
%    and plot their traces vs time
% d. plot behavior traces vs. time



%% Task 2: modeling behavior using neuronal activity
Kfolds = 10; % folds for cross validation
% a. Use the function 'train_test_regression' to train a linear regression
%    for each behavior trace based on neuronal activity using 10-fold
%    cross-validation. What is the R2? Is it "train R2"? or "Test R2"? What 
%    is the difference? 
% b. Plot each behavior trace along with the model's prediction (use hold
%    all to plot both on the same axis
% c. Use the function diffmap_euc to reduce the dimension of activity based on
%    diffusion map. Use knn-param = 100. 
[phi_x, Lambda_x1] = diffmap_euc(x, 100);
% d. Use the function plot_latent to plot the first 3 components of the 
%    Embedded activity as a scatter plot. Colors should be z-scored pupil
%    values.
% e. Use different values of knn-param and scatter plot the components 
% f. Plot the eigenvalues you got for different values of knn-param 
% g. Train a regression model for behavior based on the embedded activity
% using the first 10 components
% h. Plot behavior traces and estimated traces vs. time

%% Task 3: modeling behavior using neuronal connectivity (Euc. Distance)
% a. Write a function that extracts Pearson's correlation (use 'corr') using 
%    a sliding window. Input - x, window size (samples) and window hop (samples)
%    Outputs - tC     - connectivity values of parcels over parcels over time points. 
%              t_win  - indices of centers of analysis windows
%    Use this function to extract correlations using 3sec window size and 
%    0.1sec window hop 
% b. Use the function cmat2feat to reshape tC into a pairwise correlation
%    over time matrix
% c. Reduce the dim of c(t) using diff map (as you did for x(t))
% d. resampling behavior to match windowed c(t)
% e. Train a regression model for behavior based on the embedded
% correlations using the first 10 components
% f. Use the function plot_latent to plot the first 3 components of the
% embedded correlations as a scatter where colors are zscored pupil values
% g. Plot behavior traces and estimated traces vs. time

%% Task 5: modeling behavior using neuronal connectivity (R-Distance)

% load pre-calculated R-mean
load('../data/Rmean_animal1');
% a. Use the function getDiffMap_corr to reduce the dimension of
% correlations. To save some time, use the pre-calculated 'mRiemannianMean'
% stored in '../data/Rmean_animal1' as input. Use knn-param=100.
% b. Model (resampled) behavior using the new embedded correlations
% c. plot latent dynamics of the first 3 components with colors by pupil
% d. Plot behavior traces and estimated traces vs. time

%% Task 6
% Visualize overall modeling results

%% Task 7 - Animal 2
% Clear resuls and repeat for animal 2
clear;
close all;
data_file = '../data/data_animal2.mat';
anatomic_map_file = '../data/atlas.mat';
Kfolds = 10; % folds for cross validation
winsz = 3*fsample; % 3sec -> to samples
winhop = 1; % samples
load('../data/Rmean_animal2');

% load and plot data
load(data_file, 'x', 't', 'behavior_traces', 'behavior_labels', 'func_map');
% taking a shorter part of the experiment to reduce computation time
start_ind = 5001;
end_ind = 8e3;
x = x(:, start_ind:end_ind);
t = t(start_ind:end_ind);
behavior_traces = behavior_traces(:, start_ind:end_ind);
fsample = 10; % sampling frequency 



