clc; 
clear;

addpath('SRAD');
addpath('OBNLMpackage');

imageList = "testimgImages_simulated.txt";
outputPath = "../../outputImages/otherMethods_simulated";
% for estimating the noise level in SRAD
srad.rect = [153, 336, 156, 163];

%% filter parameters
nlm.degreeOfSmoothing = 0.075;
nlm.searchWindowsSize = 101;
nlm.comparisonWindowSize = 21;

obnlm.searchAreaSize = 101;
obnlm.patchSize = 45;
obnlm.degreeOfSmoothing = 1.05;

srad.numIterations = 200;
srad.lambda = 0.1;

median.windowSize = 15;

bilateral.degreeOfSmoothing = 0.05;
bilateral.spatialSigma = 5;

%%
imagePaths = readlines(imageList)';
if imagePaths{end} == ""
    imagePaths = imagePaths(1:end-1);
end
parfor imageIndex = 1:length(imagePaths)
    imagePath = imagePaths{imageIndex};
    otherMethodsFunc(imagePath, outputPath, nlm, obnlm, srad, median, bilateral);
end
