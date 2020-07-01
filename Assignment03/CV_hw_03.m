clc
clear all
close all

%% 1. Understanding RANSAC
% 1.1
DATA = load('LineData.mat','x','y');
x = DATA.x; y = DATA.y;
data = [x;y];% a 2xn dataset with #n data points
num = 2;% the minimum number of points. For line fitting problem, num=2
iter = 1e+5;% the number of iterations
std = 0.2;
threshDist = sqrt(3.84*(std)^2);%0.3;% the threshold of the distances between points and the fitting line
p = 0.9999;
N = log(1-p)/log(1-(1-0.5)^num);
% inlierRatio = N/length(x);
inlierRatio = 0.1;% the threshold of the number of inliers 

% Plot the data points
figure;plot(data(1,:),data(2,:),'o');hold on;

number = size(data,2); % Total number of points
bestInNum = 0; % Best fitting line with largest number of inliers
bestParameter1=0;bestParameter2=0; % parameters for best fitting line

for i=1:iter
    % Randomly select 2 points
    idx = randperm(number,num); sample = data(:,idx);   
    % Compute the distances between all points with the fitting line 
    kLine = sample(:,2)-sample(:,1);% two points relative distance
    kLineNorm = kLine/norm(kLine);
    normVector = [-kLineNorm(2),kLineNorm(1)];%Ax+By+C=0 A=-kLineNorm(2),B=kLineNorm(1)
    distance = normVector*(data - repmat(sample(:,1),1,number));
    % Compute the inliers with distances smaller than the threshold
    inlierIdx = find(abs(distance)<=threshDist);
    inlierNum = length(inlierIdx);
    % Update the number of inliers and fitting model if better model is found     
    if inlierNum>=round(inlierRatio*number) && inlierNum>bestInNum
        bestInNum = inlierNum;
        parameter1 = (sample(2,2)-sample(2,1))/(sample(1,2)-sample(1,1));
        parameter2 = sample(2,1)-parameter1*sample(1,1);
        bestParameter1=parameter1; bestParameter2=parameter2;
    end
end

% Plot the best fitting line
% figure
% xAxis = -number/2:number/2;
xAxis = -4:4;
yAxis = bestParameter1*xAxis + bestParameter2;
plot(xAxis,yAxis,'r-','LineWidth',2);

% 1.2
ADATA = load('AffineData.mat','orig_feature_pt','trans_feature_pt');
Au = ADATA.orig_feature_pt;
Av = ADATA.trans_feature_pt;
subsum = 0;
threshold = 0.10;
iter = 100;
A = zeros(6,6);
for i=1:iter
    idx = randperm(size(Au,2),3);
    
    A(1:2,:) = kron(eye(2,2),[Au(1:2,idx(1))' 1]);
    A(3:4,:) = kron(eye(2,2),[Au(1:2,idx(2))' 1]);
    A(5:6,:) = kron(eye(2,2),[Au(1:2,idx(3))' 1]);
    b = [Av(1:2,idx(1))' Av(1:2,idx(2))' Av(1:2,idx(3))'];
    % Ax = b
    x=inv(A)*b';
    
    AT(1,:) = x(1:3);
    AT(2,:) = x(4:6);
    result = AT*[Au;ones(1,size(Au,2))];
    dist = sqrt(sum((Av - result).^2));
    summation = sum(dist < threshold);
    if summation > subsum
        subsum = summation;
        outputx = x;
    end
end
Afinal = zeros(3,3);
Afinal(:,1) = outputx(1:3);
Afinal(:,2) = outputx(4:6);
Afinal(3,3) = 1;

T = maketform('Affine',Afinal);
I1 = imread('castle_original.png');
Inew = imtransform(I1, T);
imshow(Inew)
figure;
I2 = imread('castle_transformed.png');
imshow(I2)

%% 2. Distance Transforms and Chamfer Matching
clear
Im = imread('cow.png');
Im = rgb2gray(Im);
th = [0.16 0.18];
Cedge = edge(Im,'Canny',th);
imshow(Cedge)

% 2.1
cedge = double(Cedge);
cedge(find(cedge==0)) = Inf;
cedge(find(cedge==1)) = 0;
cedge0 = cedge;

for i=2:size(cedge,1)
    for j=2:size(cedge,2)
        cedge(i,j) = min([cedge(i-1,j) cedge(i,j-1) cedge(i,j)]) + 1;
    end
end

for i=size(cedge,1)-1:-1:1
    for j=size(cedge,2)-1:-1:1
        cedge(i,j) = min([cedge(i+1,j) cedge(i,j+1) cedge(i,j)]) + 1;
    end
end
% imshow(uint8(cedge))

imshow(uint8(cedge));
figure
subplot 131; imshow(uint8(bwdist(Cedge,'chessboard')));
subplot 132; imshow(uint8(bwdist(Cedge,'cityblock')));
subplot 133; imshow(uint8(bwdist(Cedge,'euclidean')));

% 2.2
cedgeTemp = cedge;
cedgeTemp = double(cedgeTemp)./256;
ImTemp = imread('template.png');
ImTemp = double(ImTemp);
for i=1:size(cedgeTemp,1)-size(ImTemp,1)
    for j=1:size(cedgeTemp,2)-size(ImTemp,2)
        Diff = ImTemp.*cedgeTemp(i:i+size(ImTemp,1)-1,j:j+size(ImTemp,2)-1);
        Dist(i,j) = sum(sum(abs(Diff)));%sqrt(sum(sum(Diff.^2)));
    end
end

[M,I] = min(Dist(:));
[I_row, I_col] = ind2sub(size(Dist),I);

img2 = uint8(cedgeTemp);
ImCow = imread('cow.png');
for i=1:size(ImTemp,1)
    for j=1:size(ImTemp,2)
        if(ImTemp(i,j)==1)
            ImCow(i+I_row-1,j+I_col-1)=255;
        end
    end
end
imshow(ImCow)
