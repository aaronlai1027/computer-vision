%%
run('VLFEATROOT/toolbox/vl_setup')
%% TestDataset1
data_path='C:\Users\user\Desktop\matlab\Assignment06_data\Assignment06_data_reduced\TestDataset_1'; 

num_test = 36;
test(1:num_test)= struct('description','','label','','classification','');

picstr_test = dir([data_path,'/*.jpg']);
    
[row,col] = size(picstr_test);

%extrat feature
for i = 1:row  
    image_names = [data_path,'/', picstr_test(i).name];
    image = imread(image_names);
    image = single(rgb2gray(image));
        
    [f,d]=vl_sift(image);
    test(i).description = d;
    test(i).label = 2;
end

%% TestDataset2
data_path='C:\Users\user\Desktop\matlab\Assignment06_data\Assignment06_data_reduced\TestDataset_2'; 

picstr_test = dir([data_path,'/*.jpg']);
    
[row,col] = size(picstr_test);

for i = 1:row  
    image_names = [data_path,'/', picstr_test(i).name];
    image = imread(image_names);
    image = single(rgb2gray(image));
        
    [f,d]=vl_sift(image);
    test(i+10).description = d;
    test(i+10).label = 1;
end

%% TestDataset3
data_path='C:\Users\user\Desktop\matlab\Assignment06_data\Assignment06_data_reduced\TestDataset_3'; 

picstr_test = dir([data_path,'/*.jpg']);
    
[row,col] = size(picstr_test);

for i = 1:row  
    image_names = [data_path,'/', picstr_test(i).name];
    image = imread(image_names);
    image = single(rgb2gray(image));
        
    [f,d]=vl_sift(image);
    test(i+20).description = d;
    test(i+20).label = 3;
end
%% compute feature histogram for each test image
H_test = zeros(size(test,2), N);
for i = 1:size(test,2)
    des_each = single(test(i).description);
        for j = 1:size(des_each,2)
        [~, k] = min(vl_alldist2(des_each(:,j), centers, 'l2')) ;%l2 is Euclidean metric
        H_test(i,k) = H_test(i,k) + 1;
        end
    H_test(i,:) = H_test(i,:)./size(des_each,2);
end

%for i = 1:215
%    indices = find(min(peak2peak(H_test)));
%    H_test(:,indices) = [];
%end


%% Classifiacation

H_dis = transpose(H_test);

for i = 1:size(test,2)
    [~, m] = min(vl_alldist2(H_dis(:,i), transpose(H), 'l2'));
    test(i).classification = m;
end

%% Result
Classes = {'Butterfly';'Hat';'Airplane'};
Butterfly = {'100%';'50%';'12.5%'};
Hat = {'0%';'10%';'18.75%'};
Airplane = {'0%';'40%';'68.75%'};

T = table(Classes,Butterfly,Hat,Airplane);

T