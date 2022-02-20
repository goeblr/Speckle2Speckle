% /* Pierrick Coupe - pierrick.coupe@gmail.com                               */
% /* Brain Imaging Center, Montreal Neurological Institute.                  */
% /* Mc Gill University                                                      */
% /*                                                                         */
% /* Copyright (C) 2008 Pierrick Coupe                                       */
%
%
%
% /*                 Details on Bayesian NLM filter                         */
% /***************************************************************************
%  *  The bayesian NLM filter is described in:                               *
%  *                                                                         *
%  * P. Coupe, P. Hellier, C. Kervrann, C. Barillot.                         *
%  * NonLocal Means-based Speckle Filtering for Ultrasound Images.           *
%  * IEEE Transactions on Image Processing, 18(10):2221-9, 2009.             *
%  ***************************************************************************/
%
%
% /*                 Details on blockwise NLM filter                        */
% /***************************************************************************
%  *  The blockwise NLM filter is described in:                              *
%  *                                                                         *
%  *  P. Coupe, P. Yger, S. Prima, P. Hellier, C. Kervrann, C. Barillot.     *
%  *  An Optimized Blockwise Non Local Means Denoising Filter for 3D Magnetic*
%  *  Resonance Images. IEEE Transactions on Medical Imaging, 27(4):425-441, *
%  *  Avril 2008                                                             *
%  ***************************************************************************/
%
%
%
% /*                 This method is patented as follows                     */
% /***************************************************************************
% * P. Coupe, P. Hellier, C. Kervrann, C. Barillot. Dispositif de traitement *
% * d'images ameliore. INRIA, Patent: 08/02206, 2008.                        *
% * Publication No. : WO/2009/133307.                                        *
% * International Application No. : PCT/FR2009/000445                        *
% *                                                                          *
%  ***************************************************************************/

function [image_filtered, speckle, image] = OBNLM(image, searchAreaSize, patchSize, degreeOfSmoothing)

assert(mod(searchAreaSize, 2) == 1);
assert(mod(patchSize, 2) == 1);

%params
M = (searchAreaSize - 1) / 2; %7;      % search area size (2*M + 1)^2
alpha = (patchSize - 1) / 2; %3;  % patch size (2*alpha + 1)^2
h = degreeOfSmoothing; % 0.7;    % smoothing parameter [0-infinite].
% If you can see structures in the "Residual image" decrease this parameter
% If you can see speckle in the "denoised image" increase this parameter

offset = 100; % to avoid Nan in Pearson divergence computation
% According to the gain used by your US device this offset can be adjusted.
                
% Intensity normalization
image = double(image);
min_input = (min(image(:)));
image = (image - min_input);
max_input = max(image(:));
image = (image / max_input) * 255;
image = image + offset; % add offset to enable the pearson divergence computation (i.e. avoid division by zero).
s = size(image);

% Padding
image = padarray(image,[alpha alpha],'symmetric');
image_filtered = bnlm2D(image,M,alpha,h);
image_filtered = image_filtered - offset;
image = image - offset;
image = image(alpha+1: s(1)+alpha, alpha+1: s(2)+alpha);
image_filtered = image_filtered(alpha+1: s(1)+alpha, alpha+1: s(2)+alpha);

% unnormalize
image = image / 255 * max_input + min_input;
image_filtered = image_filtered / 255 * max_input + min_input;
speckle = abs(image(:,:) - image_filtered(:,:));

% Display
% minds = min(image(:));
% maxds = max(image(:));
% figure;
% imagesc(image,[min_input, min_input + max_input]);
% title('Original')
% colormap(gray);
% colorbar;
% figure;
% colormap(gray);
% imagesc(image_filtered, [min_input, min_input + max_input]);
% title('Denoised by Bayesian NLM')
% colorbar;
% figure;
% colormap(gray);
% 
% imagesc(speckle);
% title('Residual image')
% colorbar;

%     image_filtered = image_filtered/ max(image(:));
%     image = image/ max(image(:)); 
%     speckle = speckle/ max(image(:)); 

% imwrite(fimg,[pathout nout]);
% imwrite(img,[pathout nameinnorm]);
%imwrite(speckle,[pathout namespeckle]);
        
end