%% Install vlfeat
run('VLFEATROOT/toolbox/vl_setup')
%% Cluster all the SIFT feature descriptors
data_path='C:/Users/user/Desktop/matlab/Assignment06_data/Assignment06_data_reduced'; 

%Extract class names
fileID = fopen('labels.txt');
class_names = textscan(fileID,'%s');
fclose(fileID);

[num_class, b] = size(class_names{1,1});
classes = class_names{1,1};

train(1:num_class)= struct('description',[],'label','');
des=[];

for i = 1:num_class
    class_name = classes{i,1};
    picstr = dir([data_path,'/TrainingDataset/', class_name, '/*.jpg']);
    
    [row,col] = size(picstr);
    
    for j = 1:row
        image_names = [data_path,'/TrainingDataset/', class_name,'/', picstr(j).name];
        image = imread(image_names);
        image = single(rgb2gray(image));
        
        [f,d]=vl_sift(image);
        des=[des,d];
    end  
    
    train(i).description=des;
    train(i).label=class_name;
    des = [];
end

%% Kmeans
N = 1000;
des = single([train(1).description,train(2).description,train(3).description]);

%Elkan.
[centers, assignments] = vl_kmeans(des, N,'algorithm', 'elkan','Initialization', 'plusplus');

%% form a histogram of N for each class

H = zeros(num_class, N);
for i = 1:num_class
    des_each = single(train(i).description);
        for j = 1:size(des_each,2)
        [~, k] = min(vl_alldist2(des_each(:,j), centers, 'l2')) ;%l2 is Euclidean metric
        H(i,k) = H(i,k) + 1;
        end
    H(i,:) = H(i,:)./size(des_each,2);
end
%indices = find(peak2peak(H)<0.0004);
%H(:,indices) = [];

subplot 311;bar(H(1,:));title(classes(1))
subplot 312;bar(H(2,:));title(classes(2))
subplot 313;bar(H(3,:));title(classes(3))


