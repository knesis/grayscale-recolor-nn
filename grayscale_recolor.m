% Estimation of Cartoon Image Color from Grayscale using HSV

% Recoloring problem modeled as regression problem by converting RGB images to HSV.
%	Target functionality - numerical mapping from grayscale value channel (V) to color hue channel (H).
%	Assumption: Color saturation channel (S) is max value due to vibrance of animated images.

%% Determination of Training and Test Sets

datapath = 'simpsons-characters-dataset/';
trainDp = 'simpsons_trainset/';
testDp = 'simpsons_testset/';

trainXp = 'simpsons_trainX/';
trainYp = 'simpsons_trainY/';
valXp = 'simpsons_valX/';
valYp = 'simpsons_valY/';
testXp = 'simpsons_testX/';
testYp = 'simpsons_testY/';

% Seed random number generator
rng(0);

% Test set pre-defined from download
testdir = dir([datapath testDp]);
num_test = length(testdir);

% Get filenames of training set images
traindir = dir([datapath, trainDp]);
train_fname = [];
for i=3:length(traindir)
    trfoldpath = [datapath trainDp traindir(i).name];
    imgdir = dir(trfoldpath);
    num_trimg = length(dir(trfoldpath));
    tridx = randi([3,num_trimg],70,1);
    
    % Form vector of file names for training set (# > 2x test set)
    for j=1:length(tridx)
        train_fname = cat(1,train_fname,{[datapath trainDp traindir(i).name '/' imgdir(tridx(j)).name]});
    end
            
end

% Get filenames of test set images
test_fname = [];
for i=3:length(testdir)
    test_fname = cat(1,test_fname,{[datapath testDp testdir(i).name]});
end

%% Data Pre-processing (run once)

% Create and save grayscale images (input)/ hue images (output) for training set
for i = 1:length(train_fname)-200
    
    % Resize images to 64x64
    img_orig = imread(train_fname{i});
    img_orig = imresize(img_orig,[64,64]);

    % Extract grayscale and hue matrices
    img_gray = im2double(rgb2gray(img_orig));
    img_hsv = rgb2hsv(img_orig);
    img_hue = img_hsv(:,:,1);
    
    save([datapath trainXp 'imgX_' num2str(i,'%04i') '.mat'],'img_gray');
    save([datapath trainYp 'imgY_' num2str(i,'%04i') '.mat'],'img_hue');
    
    clear img_orig img_gray img_hsv img_hue;
end

% Create input/output for validation set (~10% of training set)
for i=length(train_fname)-199:length(train_fname)
    
    % Same as above
    img_orig = imread(train_fname{i});
    img_orig = imresize(img_orig,[64 64]);

    img_gray = im2double(rgb2gray(img_orig));
    img_hsv = rgb2hsv(img_orig);
    img_hue = img_hsv(:,:,1);
    
    save([datapath valXp 'imgVX_' num2str(i,'%04i') '.mat'],'img_gray');
    save([datapath valYp 'imgVY_' num2str(i,'%04i') '.mat'],'img_hue');
    
    clear img_orig img_gray img_hsv img_hue;
end


% Save grayscale input and hue output images for test set
for j = 1:length(test_fname)
    
    img_origt = imread(test_fname{j});
    img_origt = imresize(img_origt,[64,64]);
    img_test = im2double(rgb2gray(img_origt));
    
    img_hsvt = rgb2hsv(img_origt);
    img_huet = img_hsvt(:,:,1);
    
    save([datapath testXp 'imgSX_' num2str(j,'%04i') '.mat'],'img_test');
    save([datapath testYp 'imgSY_' num2str(j,'%04i') '.mat'],'img_huet');
  
    clear img_origt img_test img_hsvt img_huet;
end

clear train_fname test_fname

%% Assemble Training/Validation Data

% Create vectors of training filenames
trainXf = dir([datapath trainXp]);
trainXf = [cat(1,trainXf(3:end).name)];
trainXf = [repmat([datapath trainXp],[size(trainXf,1),1]) trainXf];

trainYf = dir([datapath trainYp]);
trainYf = [cat(1,trainYf(3:end).name)];
trainYf = [repmat([datapath trainYp],[size(trainYf,1),1]) trainYf];

valXf = dir([datapath valXp]);
valXf = [cat(1,valXf(3:end).name)];
valXf = [repmat([datapath valXp],[size(valXf,1),1]) valXf];

valYf = dir([datapath valYp]);
valYf = [cat(1,valYf(3:end).name)];
valYf = [repmat([datapath valYp],[size(valYf,1),1]) valYf];


% Load training data into memory for network training (not the best approach)
X = []; Y = [];
for i=1:size(trainXf,1)
    imgx = load(trainXf(i,:)); imgx = imgx.img_gray;
    imgy = load(trainYf(i,:)); imgy = imgy.img_hue;
    X(:,:,1,i) = imgx;
    Y(:,:,1,i) = imgy;
end

VX = []; VY=[];
for i=1:size(valXf,1)
    vix = load(valXf(i,:)); vix = vix.img_gray;
    viy = load(valYf(i,:)); viy = viy.img_hue;
    VX(:,:,1,i) = vix;
    VY(:,:,1,i) = viy;
end

save('X.mat','X');
save('Y.mat','Y');
save('VX.mat','VX');
save('VY.mat','VY');

%% Build and Train Network

% Load prestored data
load('X.mat','X');
load('Y.mat','Y');
load('VX.mat','VX');
load('VY.mat','VY');

% Standardize data by training set
meanX = mean(X,4);
stdX = std(X,[],4);

Xn = (X-meanX)./stdX;
Yn = (Y-meanX)./stdX;
VXn = (VX-meanX)./stdX;
VYn = (VY-meanX)./stdX;


% Define convolutional network structure
layers= [imageInputLayer([64 64 1],'normalization','none')
    convolution2dLayer(3,8,'Padding','same')
    reluLayer
    convolution2dLayer(3,16,'Padding','same');
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,32,'Padding','same');
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,1,'Padding','same');
    regressionLayer];

% Set training options (RMS propagation loss function)
options = trainingOptions('rmsprop', ...
    'MaxEpochs',20, ...
    'InitialLearnRate',1e-1, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{VXn,VYn},...
    'ValidationFrequency',1);

% Train network
[net,info] = trainNetwork(Xn,Yn,layers,options);

% Store configured network and training statistics
save('color_net.mat','net');
save('net_info.mat','info');

% Plot training info
load('net_info.mat');
figure; plot(info.TrainingRMSE,'LineWidth',2); grid on; hold on;
plot(info.ValidationRMSE,'LineWidth',2);
xlabel('Iterations'); ylabel('RMSE');
title('Training and Validation RMSE for Colorization Net');
legend('Training Set','Validation Set','Location','ne');


%% Test Network

% Create vectors of test filenames
testXf = dir([datapath testXp]);
testXf = [cat(1,testXf(3:end).name)];
testXf = [repmat([datapath testXp],[size(testXf,1),1]) testXf];

testYf = dir([datapath testYp]);
testYf = [cat(1,testYf(3:end).name)];
testYf = [repmat([datapath testYp],[size(testYf,1),1]) testYf];


% Load test data and normalize
SX = []; SY = [];
for i=1:size(testXf,1)
    timg = load(testXf(i,:)); timg = timg.img_test;
    thue = load(testYf(i,:)); thue = thue.img_huet;
    SX(:,:,1,i) = timg;
    SY(:,:,1,i) = thue;
end
SXn = (SX-meanX)./stdX;

% Test network
load('color_net.mat');
SYn = predict(net,SXn);

% Un-normalize data for similar scale to ground truth
SY_pred = (SYn.*stdX)+meanX;

% Calculate RMSE of test data (no K-means)
hue_err = reshape(sqrt(mean(mean((SY-SY_pred).^2,1),2)),[1 size(SY_pred,4)]);
hue_stat = [mean(hue_err), std(hue_err)];

%% Post-processing (K-Means)

% For images in paper
% i=501; % Top sample
% i=440; % Bottom sample

for i=1:size(SY_pred,4)
    % Utilize full range of hue value (Map [min,max] to [0,1])
    hue = SY_pred(:,:,:,i);
    
    % Perform color clustering on hue values to 10 colors
    [idx,colors] = imsegkmeans(hue,10,'NumAttempts',3);
    
    % To reduce heterogeneity, assign cluster members to mean color
    hl = colors(idx);
    imshow(hsv2rgb(cat(3,hl,ones(size(hue)),SX(:,:,:,i))));
    HY(:,:,:,i) = hl;
end

% Compute new RMSE statistics for clustered data
km_err = reshape(sqrt(mean(mean((SY-HY).^2,1),2)),[1 size(HY,4)]);
km_stat = [mean(km_err), std(km_err)];
