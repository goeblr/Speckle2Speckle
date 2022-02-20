clc; 
clear;

addpath('SRAD');
addpath('OBNLMpackage');

imagePath = "../../data/phantom.png";
outputPath = "../../outputImages/otherMethods_experimental";
% for estimating the noise level in SRAD
srad.rect = [349, 348, 141, 142];

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
otherMethodsFunc(imagePath, outputPath, nlm, obnlm, srad, median, bilateral);
