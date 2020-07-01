%% Exercise2-a Noise Reduction
cam = imread('/Users/aaron/Desktop/Computer Vision/Assignment02_546/cameraman.tif');

camp = padarray(cam,[2 2],'replicate');
filter = [2 4 5 4 2;
          4 9 12 9 4;
          5 12 15 12 5;
          4 9 12 9 4;
          2 4 5 4 2].*1/159;
nr = imfilter(double(camp),filter);
camnr = nr([3:258],[3:258]);


subplot 121, imshow(uint8(cam));
subplot 122, imshow(uint8(camnr));
%% Exercise2-b Gradient Magnitude and Angle
camnrp = padarray(camnr,[1 1],'replicate');
dx = imfilter(double(camnrp), [-1 0 1; -2 0 2; -1 0 1]);
dy = imfilter(double(camnrp), [1 2 1; 0 0 0 ; -1 -2 -1]);
dp = sqrt((dx.^2 + dy.^2));
d = dp([2:257],[2:257]);

theta = atan(dy(2:257,2:257)./dx(2:257,2:257)).*180./pi;
t = zeros(size(theta));
t(find(theta >= 67.5 | theta < -67.5)) = 90;
t(find(theta >= 22.5 & theta < 67.5)) = 45;
t(find(theta >= -22.5 & theta < 22.5)) = 0;
t(find(theta >= -67.5 & theta < -22.5)) = 135;

imshow(uint8(d));
%% %% Exercise2-c Non-Maximum Suppression
dp = padarray(d,[1 1],'replicate');
sup = zeros(size(d));
[r,c] = size(dp);

for i = 2:r-1,
    for j = 2:c-1,
        if t(i-1,j-1) == 0,
            if dp(i,j) > dp(i,j-1) & dp(i,j) > dp(i,j+1),
                sup(i-1,j-1) = dp(i,j);
            end
        elseif t(i-1,j-1) == 45,
            if dp(i,j) > dp(i-1,j+1) & dp(i,j) > dp(i+1,j-1),
                sup(i-1,j-1) = dp(i,j);
            end
        elseif t(i-1,j-1) == 90,
            if dp(i,j) > dp(i-1,j) & dp(i,j) > dp(i+1,j),
                sup(i-1,j-1) = dp(i,j);
            end
        elseif t(i-1,j-1) == 135,
            if dp(i,j) > dp(i-1,j-1) & dp(i,j) > dp(i+1,j+1),
                sup(i-1,j-1) = dp(i,j);
            end
        end
    end
end

imshow(uint8(sup));

%% Hysteresis Thresholding
th = 187;
tl = 110;

supp = padarray(sup,[2 2],'replicate');
edge = zeros(size(sup));

[r1,c1] = size(supp);

for i = 3:r1-2
    for j = 3:c1-2
        if supp(i,j) >= th,
            edge(i-2,j-2) = supp(i,j);
        elseif supp(i,j) >= tl  
            if max(max(supp(i-2:i+2,j-2:j+2))) >= th
                edge(i-2,j-2) = supp(i,j);
            end
        end
    end
end


% subplot 121, imshow(cam);
% subplot 122, imshow(uint8(edge));
imshow(uint8(edge));
%% Hybrid Images
kobe = imread('/Users/aaron/Desktop/Computer Vision/Assignment02_546/kobe.png');
aaron = imread('/Users/aaron/Desktop/Computer Vision/Assignment02_546/aaron.jpg');

k = double(rgb2gray(kobe));
a = double(rgb2gray(aaron));

k1 = imresize(k, 0.85, 'bicubic');
k1 = k1(1:end-130,1:end-30);
k2 = zeros(size(a,1),size(a,2));

k2(1+125:size(k1,1)+125, end-size(k1,2)+1:end) = k1;

L = 20;
gaussf = fspecial('gaussian',L,3);
fAP = zeros(L,L);
fAP(round(15/2),round(15/2)) = 1;
fHP = fAP - gaussf;

imAP = imfilter(k2,fAP,'symmetric');
imLP = imfilter(a,gaussf,'symmetric');
imHP = imfilter(k2,fHP,'symmetric');

im = imLP + imHP;

% subplot 121, imshow(kobe);
% subplot 122, imshow(aaron);
imshow(uint8(im))
