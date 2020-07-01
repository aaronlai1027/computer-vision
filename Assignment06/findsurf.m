%%Surf 
%%Cluster all the SIFT feature descriptors

data_path='C:\Users\user\Desktop\matlab\Assignment06_data\Assignment06_data_expanded'; 

%Extract class names
fileID = fopen('labels3.txt');
class_names = textscan(fileID,'%s');
fclose(fileID);

[num_class, b] = size(class_names{1,1});
classes = class_names{1,1};

train2(1:num_class)= struct('description',[],'label','');
des=[];

for i = 1:num_class
    class_name = classes{i,1};
    picstr = dir([data_path,'\TrainingDataset\', class_name, '\*.jpg']);
    
    [row,col] = size(picstr);
    
    for j = 1:row  
        image_names = [data_path,'\TrainingDataset\', class_name,'\', picstr(j).name];
        image = imread(image_names);
        if size(image,3) == 3
            image = rgb2gray(image);
        end
        temp =  detectSURFFeatures(image);
        [im_features, temp] = extractFeatures(image, temp);
        des = [des,transpose(im_features)];
    end  
    
    train2(i).description=des;
    train2(i).label=class_name;
    des = [];
end

%% Kmeans
N = 5000;
des = single([train2(1).description,train2(2).description,train2(3).description]);

%Elkan.
[centers, assignments] = vl_kmeans(des, N,'algorithm', 'elkan','Initialization', 'plusplus');

%% form a histogram of N for each class

H = zeros(num_class, N);
for i = 1:num_class
    des_each = single(train2(i).description);
        for j = 1:size(des_each,2)
        [~, k] = min(vl_alldist2(des_each(:,j), centers, 'l2')) ;%l2 is Euclidean metric
        H(i,k) = H(i,k) + 1;
        end
    H(i,:) = H(i,:)./size(des_each,2);
end
%indices = find(peak2peak(H)<0.0002);
%H(:,indices) = [];

subplot 511;bar(H(1,:));title(classes(1))
subplot 512;bar(H(2,:));title(classes(2))
subplot 513;bar(H(3,:));title(classes(3))
subplot 514;bar(H(4,:));title(classes(4))
subplot 515;bar(H(5,:));title(classes(5))


