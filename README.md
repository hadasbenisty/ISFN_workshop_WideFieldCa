# ISFN_workshop_WideFieldCa
A project for modeling behavior bases on neuronal activity and connectivity

This project includes data collected by Higley Lab (Yale University) and code for modeling sponteneous behavior based on neuronal signals.

# Data 
Imaging data (two animals) is stored in mat files (data/data_animal1.mat, data/data_animal2.mat) where:
* x - brain parcels X T matrix of traces of neuronal activity related to functional parcels. 
* t - T X 1 time trace
* func_map - a 256X256 matrix of the animal's brain indicating the location of each brain parcel. An area number i in this map corresponds to the activity of the i-th trace in x
* Behavior_traces - 2 X T, time traces of sponteneous behavior 
* Behaiovr_labels - 2 X 1 indicating which behavior traces are stored 

data/atlas.mat - stores the anatomic parcellation map (by the Allen institute for brain research)

Pre-processed R-mean - evaluated in advanced stored in data/Rmean_animal1.mat and data/Rmean_animal2.mat

# Code
* Main script for workshop - main.m
* 
