function otherMethodsFunc(imagePath, outputPath, nlm, obnlm, srad, median, bilateral)

    I = imread(imagePath);
    I = single(I) / 255;

    Inlm = imnlmfilt(I, 'DegreeOfSmoothing', nlm.degreeOfSmoothing, 'SearchWindowSize', nlm.searchWindowsSize, ...
         'ComparisonWindowSize', nlm.comparisonWindowSize);

    Iobnlm = OBNLM(I, obnlm.searchAreaSize, obnlm.patchSize, obnlm.degreeOfSmoothing);

    Israd = SRAD(I, srad.numIterations, srad.lambda, srad.rect);

    Imed = medfilt1(I, median.windowSize);
    Ibil = imbilatfilt(I, bilateral.degreeOfSmoothing, bilateral.spatialSigma);

    I = min(max(I, 0), 1);
    Inlm = min(max(Inlm, 0), 1);
    Iobnlm = min(max(Iobnlm, 0), 1);
    Israd = min(max(Israd, 0), 1);
    Imed = min(max(Imed, 0), 1);
    Ibil = min(max(Ibil, 0), 1);

    Rnlm = abs(I - Inlm);
    Robnlm = abs(I - Iobnlm);
    Rsrad = abs(I - Israd);
    Rmed = abs(I - Imed);
    Rbil = abs(I - Ibil);

    %% save
    [~, imageFilename, imageExtension] = fileparts(imagePath);

    imwrite(I,      fullfile(outputPath, imageFilename + "_CompIn" + imageExtension));
    imwrite(Inlm,   fullfile(outputPath, imageFilename + "_NLM"    + imageExtension));
    imwrite(Iobnlm, fullfile(outputPath, imageFilename + "_OBNLM"  + imageExtension));
    imwrite(Israd,  fullfile(outputPath, imageFilename + "_SRAD"   + imageExtension));
    imwrite(Imed,   fullfile(outputPath, imageFilename + "_MED"    + imageExtension));
    imwrite(Ibil,   fullfile(outputPath, imageFilename + "_BILAT"  + imageExtension));


    imwrite(Rnlm,   fullfile(outputPath, imageFilename + "_resNLM"    + imageExtension));
    imwrite(Robnlm, fullfile(outputPath, imageFilename + "_resOBNLM"  + imageExtension));
    imwrite(Rsrad,  fullfile(outputPath, imageFilename + "_resSRAD"   + imageExtension));
    imwrite(Rmed,   fullfile(outputPath, imageFilename + "_resMED"    + imageExtension));
    imwrite(Rbil,   fullfile(outputPath, imageFilename + "_resBILAT"  + imageExtension));
end