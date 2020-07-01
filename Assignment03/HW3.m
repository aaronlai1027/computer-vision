%% Problem1-1
data=load('/Users/aaron/Desktop/Computer Vision/HW3/Ransac/LineData.mat');% a 2xn dataset with #n data points

num =2 ;%: the minimum number of points. For line fitting problem
iter = 100 ;% the number of iterations
threshDist = 0.4;% the threshold of the distances between points and the fitting line
inlierRatio = 0.5;%: the threshold of the number of inliers 

number = length(data.x);
bestInNum = 0; % Best fitting line with largest number of inliers
bestParameter1=0;bestParameter2=0; % parameters for best fitting line
DATA = [data.x;data.y];
for i = 1:iter
    %Randomly select 2 points
    idx = randperm(number,2);
    sample = DATA(:,idx);
    % Compute the distances between all points with the fitting line 
    kLine = sample(:,2)-sample(:,1);% two points relative distance
    kLineNorm = kLine/norm(kLine);
    normVector = [-kLineNorm(2),kLineNorm(1)];%Ax+By+C=0 A=-kLineNorm(2),B=kLineNorm(1)
    distance = normVector*(DATA - repmat(sample(:,1),1,number));
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
 
figure;plot(data.x,data.y,'o');hold on;
xAxis = -4:4; 
yAxis = bestParameter1*xAxis + bestParameter2;
plot(xAxis,yAxis,'r-','LineWidth',2);

%% Problem1-2
data1 = load('/Users/aaron/Desktop/Computer Vision/HW3/Ransac/AffineData.mat.mat','orig_feature_pt','trans_feature_pt');

au = data1.orig_feature_pt;
av = data1.trans_feature_pt;
subsum = 0;
threshold = 0.10;
iter = 100;
p = 0.99;
N = log(1-p)/log(1-(1-0.6)^num);


a = zeros(6,6);

for i=1:iter
    idx = randperm(size(au,2),3);
    
    a(1:2,:) = kron(eye(2,2),[au(1:2,idx(1))' 1]);
    a(3:4,:) = kron(eye(2,2),[au(1:2,idx(2))' 1]);
    a(5:6,:) = kron(eye(2,2),[au(1:2,idx(3))' 1]);
    b = [av(1:2,idx(1))' av(1:2,idx(2))' av(1:2,idx(3))'];
    
    % Ax = b
    x=inv(a)*b';
    
    AT(1,:) = x(1:3);
    AT(2,:) = x(4:6);
    result = AT*[au;ones(1,size(au,2))];
    dist = sqrt(sum((av - result).^2));
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







%% Problem2-1-1
cow = imread('/Users/aaron/Desktop/Computer Vision/HW3/DistanceTransform_ChamferMatching/cow.png');
grcow = rgb2gray(cow);
edgecow = edge(grcow, 'canny', [0.14,0.16]);
imshow(edgecow)

%% Problem2-1-2
d = zeros(size(edgecow));
d(find(edgecow == 0)) = Inf;
d(find(edgecow == 1)) = 0;

[r,c] = size(edgecow);

%forward
for i = 2:r
    for j = 2:c
        d(i,j) = min([d(i,j),d(i,j-1)+1,d(i-1,j)+1]);
    end
end

%backward
for i = fliplr(1:r-1)
    for j = fliplr(1:c-1)
        d(i,j) = min([d(i,j),d(i,j+1)+1,d(i+1,j)+1]);
    end
end
imshow(uint8(d))
%% Problem2-1-3
subplot 221, imshow(edgecow); title('edge');
subplot 222, imshow(uint8(bwdist(edgecow,'euclidean'))), title('euclidean');
subplot 223, imshow(uint8(bwdist(edgecow,'cityblock'))), title('cityblock');
subplot 224, imshow(uint8(bwdist(edgecow,'chessboard'))), title('chessboard');

%% Problem2-2-1

tem = imread('/Users/aaron/Desktop/Computer Vision/HW3/DistanceTransform_ChamferMatching/template.png');

r1 = size(d,1) - size(tem,1) + 1;
c1 = size(d,2) - size(tem,2) + 1;
diff = zeros(r1,c1);
for i = 1:r1
    for j = 1:c1
        diff(i,j) = sum(sum(abs(d(i:i+size(tem,1)-1,j:j+size(tem,2)-1)./256.*tem)));
        charmd = min(min(diff));
    end
end

[r2,c2]=min(diff(:));
[r3,c3]=ind2sub(size(diff),c2)

for i = 1:size(tem,1)
    for j = 1:size(tem,2)
        if (tem(i,j)==1)
            cow(i+r3,j+c3)=255;
        end
    end
end
imshow(cow)
