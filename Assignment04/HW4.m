%% Problem1 
% Harris Corner Detection

% ?Cornerness? measure
im = imread('chessboard.jpg');

%Change background to white
imr = imrotate(im,30); %Rotate the image
mrot = ~imrotate(true(size(im)),30);
imr(mrot ~= zeros(size(mrot))) = 255;

ims = imresize(im,4); %Resize the image

figure(1)
harris(im, 2, 3, 2000) %original image
figure(2)
harris(imr, 2, 3, 5000) %rotated image
figure(3)
harris(ims, 8, 5, 1000) %Resized image

function harris(image, sigma, radius, threshold)
%convert to gray image
grc = rgb2gray(image);

%Derivates of x and y
[dx,dy] = meshgrid(-1:1,-1:1);
ix = imfilter(double(grc),dx,'replicate');
iy = imfilter(double(grc),dy,'replicate');

%Create gassian filter
s = 2;
g = fspecial('gaussian',4*s,s);

%Apply to Harris matrix
ix2 = imfilter(ix.^2,g,'replicate');
iy2 = imfilter(iy.^2,g,'replicate');
ixy = imfilter(ix.*iy,g,'replicate');

%Compute the cornerness measure M
M = (ix2.*iy2-ixy.^2)./(ix2+iy2 + eps);

% Corner extraction
r = radius;
thresh = threshold;

%Perform non-maximal suppression
Ms = ordfilt2(M,r^2,true(r));
corpoints = (M == Ms) & (M > thresh);

%Find the coordinates of the corner points
[row,col] = find(corpoints);

%Display the image and superimpose the corners
imshow(image),hold on,
plot(col,row,'ys');
end








