# ISFN_workshop_WideFieldCa
A project for modeling behavior bases on neuronal activity and connectivity

This project includes data collected by Higley Lab (Yale University) and code for modeling sponteneous behavior based on neuronal signals.

## Data 
Imaging data (two animals) is stored in mat files (data/data_animal1.mat, data/data_animal2.mat) where:
* x - brain parcels X T matrix of traces of neuronal activity related to functional parcels. 
* t - T X 1 time trace
* func_map - a 256X256 matrix of the animal's brain indicating the location of each brain parcel. An area number i in this map corresponds to the activity of the i-th trace in x
* Behavior_traces - 3 X T, time traces of sponteneous behavior 
* Behaiovr_labels - 3 X 1 indicating which behavior traces are stored 

data/atlas.mat - stores the anatomic parcellation map (by the Allen institute for brain research)

Pre-processed R-mean - evaluated in advanced stored in data/Rmean_animal1.mat and data/Rmean_animal2.mat

## Code
For workshop particiapnts please use the code in 'workshop_code', main script for workshop - main.m
 

Full solutions and missing functions are located in the folder 'code'.

## Background Material
* A short summary on widefield Ca imaging and functional parcellation (chapter 5) - https://www.spiedigitallibrary.org/journalArticle/Download?urlId=10.1117%2F1.NPh.9.4.041402
* Diffusion map (wiki) - https://en.wikipedia.org/wiki/Diffusion_map
* Dimensionality reduction of correlation matrices based on diffusion map (See Materials and Methods section) - https://www.biorxiv.org/content/10.1101/2021.08.15.456390v3.full.pdf


## Project
### Task A: visualize data
1. plot the anatomic parcellation map saved in '../data/atlas.mat'
2. plot the functional parcellation map stored as func_map
3. choose the functional parcel corresponding to anatomical visual cortex (anatomical #1) and corresponding to the motor cortex (anatomical #23) and plot their traces vs time
4. plot behavior traces vs. time



### Task B: modeling behavior using neuronal activity
1. Use the function 'train_test_regression' to train a linear regression for each behavior trace based on neuronal activity using 10-fold cross-validation. What is the R2? Is it "train R2"? or "Test R2"? What is the difference? 
2. Plot each behavior trace along with the model's prediction (use hold all to plot both on the same axis 
3. Use the function diffmap_euc to reduce the dimension of activity based on diffusion map. Use knn-param = 100. 
4. Use the function plot_latent to plot the first 3 components of the embedded activity as a scatter plot. Colors should be z-scored pupil values.
5. Use different values of knn-param and scatter plot the components 
6. Plot the eigenvalues you got for different values of knn-param 
7. Train a regression model for behavior based on the embedded activity using the first 10 components
8. Plot behavior traces and estimated traces vs. time

### Task C: modeling behavior using neuronal connectivity (Euc. Distance)
1. Write a function that extracts Pearson's correlation (use 'corr') using a sliding window. Input - x, window size (samples) and window hop (samples)
   Outputs - tC     - connectivity values of parcels over parcels over time points. 
             t_win  - indices of centers of analysis windows
   Use this function to extract correlations using 3sec window size and 0.1sec window hop 
2. Use the function cmat2feat to reshape tC into a pairwise correlation over time matrix
3. Reduce the dim of c(t) using diff map (as you did for x(t))
4. resampling behavior to match windowed c(t)
5. Train a regression model for behavior based on the embedded correlations using the first 10 components
6. Use the function plot_latent to plot the first 3 components of the embedded correlations as a scatter where colors are zscored pupil values
7. Plot behavior traces and estimated traces vs. time

### Task D: modeling behavior using neuronal connectivity (R-Distance)


1. Use the function getDiffMap_corr to reduce the dimension of correlations. To save some time, use the pre-calculated 'mRiemannianMean'stored in '../data/Rmean_animal1' as input. Use knn-param=100.
2. Model (resampled) behavior using the new embedded correlations
3. plot latent dynamics of the first 3 components with colors by pupil
4. Plot behavior traces and estimated traces vs. time

### Task E
Visualize overall modeling results

### Task F - Animal 2
Clear resuls and repeat for animal 2
end_ind);
fsample = 10; % sampling frequency 




