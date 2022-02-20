clear all, close all

% This example performs speckle reducing anisotropic diffusion on
% ultrasound phantom image of a elliptical object via the function SRAD.
% SRAD is implemented following:
%   Y. Yu and S.T. Acton, "Speckle reducing anisotropic diffusion," IEEE
%   Transactions on Image Processing, vol. 11, pp. 1260-1270, 2002.
%   <http://viva.ee.virginia.edu/publications/j_srad.pdf>

%WRITTEN BY:  Drew Gilliam
%
%MODIFICATION HISTORY:
%   2006.03   Drew Gilliam
%       --creation


% file name
filename = 'ultrasound phantom.bmp';

% read in file, make double
I = im2double( imread(filename) );

% region of uniform speckle
% NOTE if rect is empty (i.e. "rect = []"), the user can choose an
% arbitrary region of uniform speckle at runtime.  The original image is
% displayed on the current axes, and the user may click and drag a
% rectangle to define the uniform speckle region
rect = [10 15 40 40];   % default rectangle
% rect = [];              % choose rectangle at runtime

% SRAD image
[J,rect] = SRAD(I,200,0.5,rect);

% plot both images
figure
subplot(1,2,1)
imshow(I,[])
subplot(1,2,2)
imshow(J,[])



%**************************************************************************
% END OF FILE
%**************************************************************************