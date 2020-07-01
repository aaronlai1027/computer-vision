echo off
clear all
home
echo on

%% PART1
%% 
%Exercise1-1
im1 = imread('/Users/aaron/Desktop/Computer Vision/Assignment 01/ironman.jpg');
headim = im1(1:200, 230:400, :);

pause;
%% 
%Exercise1-2
imwrite(headim, 'head.jpg');
imshow(headim)
pause;
%% 
%Exercise1-3
ghead = headim;
ghead(:,:,1) = 0; 
ghead(:,:,3) = 0; 
imshow(ghead);

pause;
%% 
%Exercise1-4
imshow(im1(:,:,[2,1,3]));

pause;
%% 
%Exercise2-1
im2 = imread('/Users/aaron/Desktop/Computer Vision/Assignment 01/barbara.jpg');
grim2 = rgb2gray(im2);
imshow(grim2)
pause;
%% 
%Exercise2-2
imhist(grim2,5);

pause;
%% 
%Exercise2-3
gaussFilt2 = fspecial('gaussian',15,2);
gaussFilt8 = fspecial('gaussian',15,8);

convImgr2 = imfilter(grim2, gaussFilt2, 'symmetric'); 
convImgr8 = imfilter(grim2, gaussFilt8, 'symmetric'); 

%% 
%Exercise2-4
subplot 311, imhist(convImgr8), title('Gaussian SD=8')
subplot 312, imhist(convImgr2), title('Gaussian SD=2')
subplot 313, imhist(im2), title('Original Image')

pause;
%% 
%Exercise2-5
convIm2 = imfilter(im2, gaussFilt2, 'symmetric'); 
imshow(convIm2-im2);

pause;
%% 
%Exercise2-6
re = convIm2-im2;
thresh = max(re(:))*0.05
tim2 = re;
index = find(re<=thresh);
tim2(index) = 0;
%% 
%Exercise2-7
imshow(tim2);

pause;
%% PART2
%% 
%Filtering
I1 = [120 110 90  115 40;
      145 135 135 65  35;
      125 115 55  35  25;
      80  45  45  20  15;
      40  35  25  10  10]

I2 = [125 130 135 110 125;
      145 135 135 155 125;
      65  60  55  45  40;
      40  35  40  25  15;
      15  15  20  15  10]

PI1 = padarray(I1, [1,1], 'replicate')
PI2 = padarray(I2, [1,1], 'replicate')


%% 
%Filter1

I1_filter1 = zeros(5,5)

[rows1, cols1] = size(PI1);
for i = 1:rows1,
    for j = 1:cols1,
        if i > 1 & i < rows1 & j > 1 & j < cols1,
            I1_filter1(i-1,j-1) = (PI1(i,j)+PI1(i,j-1)+PI1(i,j+1))/3;
            
        end
    end
end

I2_filter1 = zeros(5,5)

[rows2, cols2] = size(PI2);
for i = 1:rows2,
    for j = 1:cols2,
        if i > 1 & i < rows2 & j > 1 & j < cols2,
            I2_filter1(i-1,j-1) = (PI2(i,j)+PI2(i,j-1)+PI2(i,j+1))/3;
            
        end
    end
end

I1_filter1
I2_filter1 

pause;
%% 
%filter2

I1_filter2 = zeros(5,5);

[rows1, cols1] = size(PI1);
for i = 1:rows1,
    for j = 1:cols1,
        if i > 1 & i < rows1 & j > 1 & j < cols1,
            I1_filter2(i-1,j-1) = (PI1(i-1,j)+PI1(i,j)+PI1(i+1,j))/3;
        end
    end
end

I2_filter2 = zeros(5,5);

[rows2, cols2] = size(PI2);
for i = 1:rows2,
    for j = 1:cols2,
        if i > 1 & i < rows2 & j > 1 & j < cols2,
            I2_filter2(i-1,j-1) = (PI2(i-1,j)+PI2(i,j)+PI2(i+1,j))/3;
        end
    end
end

I1_filter2
I2_filter2

pause;
%% 
%filter3

I1_filter3 = zeros(5,5);

[rows1, cols1] = size(PI1);
for i = 1:rows1,
    for j = 1:cols1,
        if i > 1 & i < rows1 & j > 1 & j < cols1,
            I1_filter3(i-1,j-1) = (PI1(i-1,j-1)+PI1(i-1,j)+PI1(i-1,j+1)+PI1(i,j-1)+PI1(i,j)+PI1(i,j+1)+PI1(i+1,j-1)+PI1(i+1,j)+PI1(i+1,j+1))/9;
        end
    end
end

I2_filter3 = zeros(5,5);

[rows2, cols2] = size(PI2);
for i = 1:rows2,
    for j = 1:cols2,
        if i > 1 & i < rows2 & j > 1 & j < cols2,
            I2_filter3(i-1,j-1) = (PI2(i-1,j-1)+PI2(i-1,j)+PI2(i-1,j+1)+PI2(i,j-1)+PI2(i,j)+PI2(i,j+1)+PI2(i+1,j-1)+PI2(i+1,j)+PI2(i+1,j+1))/9;
        end
    end
end


I1_filter3
I2_filter3

%% 
%Central difference Gradient filter
dx = imfilter(double(grim2)/256, [-1/2 0 1/2]);
dy = imfilter(double(grim2)/256, [-1/2 0 1/2]');
cenbar = (dx.^2 + dy.^2).^(1/2);

%% 
%Sobel filter
sobfilt = fspecial('sobel');
sobbar = imfilter(grim2,sobfilt);

%% 
%Mean filter
meanfilt = fspecial('average');
meanbar = imfilter(grim2,meanfilt);

%% 
%Median filter
medbar = medfilt2(grim2);

subplot 221,imshow(cenbar), title('Central difference Gradient filter')
subplot 222,imshow(sobbar), title('Sobel filter')
subplot 223,imshow(meanbar), title('Mean filter')
subplot 224,imshow(medbar), title('Median filter')

pause;

%% 
%smoothing_average
im3 = imread('/Users/aaron/Desktop/Computer Vision/Assignment 01/camera_man_noisy.png');
avfilt2 = fspecial('average',[2 2]);
avfilt4 = fspecial('average',[4 4]);
avfilt8 = fspecial('average',[8 8]);
avfilt16 = fspecial('average',[16 16]);

av2cam = imfilter(im3,avfilt2);
av4cam = imfilter(im3,avfilt4);
av8cam = imfilter(im3,avfilt8);
av16cam = imfilter(im3,avfilt16);

subplot 221,imshow(av2cam), title('Average filter size 2x2')
subplot 222,imshow(av4cam), title('Average filter size 4x4')
subplot 223,imshow(av8cam), title('Average filter size 8x8')
subplot 224,imshow(av16cam), title('Average filter size 16x16')

pause;
%% 
%smoothing_gaussan

gafilt2 = fspecial('gaussian',8,2);
gafilt4 = fspecial('gaussian',16,4);
gafilt8 = fspecial('gaussian',32,8);
gafilt16 = fspecial('gaussian',64,16);

ga2cam = imfilter(im3,gafilt2);
ga4cam = imfilter(im3,gafilt4);
ga8cam = imfilter(im3,gafilt8);
ga16cam = imfilter(im3,gafilt16);

subplot 221,imshow(ga2cam), title('Gaussian filter SD=2')
subplot 222,imshow(ga4cam), title('Gaussian filter SD=4')
subplot 223,imshow(ga8cam), title('Gaussian filter SD=8')
subplot 224,imshow(ga16cam), title('Gaussian filter SD=16')

pause;
%% 
%Edge preserving smoothing

cam = double(im3)/255;

% Set bilateral filter parameters.
w     = 5;       % bilateral filter half-width
sigma = [3 0.35]; % bilateral filter standard deviations

% Apply bilateral filter to each image.
bflt_cam = bfilter2(cam,w,sigma);

imagesc(bflt_cam);axis image; colormap gray;





