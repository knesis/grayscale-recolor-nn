Recoloring Grayscale Images using Neural Networks


The files included for this project are as follows:

grayscale_recolor.m - Main script for this project, separated into sections pertaining to the following tasks:
	Selecting Training/Test Set
		Randomly defines indices for training and test sets, parses into vector of filenames
	Data Pre-processing
		Takes color images from dataset and saves input data (grayscale) and output ground truth data (hue channel) at 64x64 size into specific directories
	Loading Data into Workspace
		Uses data directories to load training and validation data into memory (max size = 38 Mb)
	Building/Training Network
		Builds network layers and trains using train/validation data
	Testing Network
		Loads network model saved in previous section and evaluates on test data (loaded into memory from directories)
		Model file 'color_net.mat' should be in working directory
	Data Post-processing
		Implements K-means post-processing and measures RMSE statistics for output predictions
		
Data set extracted from Kaggle Simpsons character dataset:
https://www.kaggle.com/alexattia/the-simpsons-characters-dataset
		
		

Colorization code is implemented in MATLAB R2018b and uses the following libraries:
	Image Processing Toolbox
	Deep Learning Toolbox