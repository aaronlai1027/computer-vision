%%
run('VLFEATROOT/toolbox/vl_setup')
%% TestDataset1
data_path='C:\Users\user\Desktop\matlab\Assignment06_data\Assignment06_data_expanded\TestDataset'; 

num_test = 143;
test2(1:num_test)= struct('description','','label','','classification','');

picstr_test = dir([data_path,'/*.jpg']);
    
[row,col] = size(picstr_test);

%extrat feature
for i = 1:row  
    image_names = [data_path,'/', picstr_test(i).name];
    image = imread(image_names);
    if size(image,3) == 3
        image = rgb2gray(image);
    end
        
    temp =  detectSURFFeatures(image);
    [im_features, temp] = extractFeatures(image, temp);
    test2(i).description = transpose(im_features);
    if i<13
    test2(i).label = 1;
    elseif i<27
    test2(i).label = 2;
    elseif i<37
    test2(i).label = 3;
    elseif i<47
    test2(i).label = 4;
    elseif i<61
    test2(i).label = 5;
    elseif i<71
    test2(i).label = 6;
    elseif i<81
    test2(i).label = 7;
    elseif i<91
    test2(i).label = 8;
    elseif i<101
    test2(i).label = 9;
    elseif i<116
    test2(i).label = 10;
    elseif i<130
    test2(i).label = 11;
    elseif i<144
    test2(i).label = 12;
%     elseif i<154
%     test2(i).label = 13;
%     elseif i<164
%     test2(i).label = 14;
%     elseif i<174
%     test2(i).label = 15;
%     elseif i<184
%     test2(i).label = 16;
%     elseif i<194
%     test2(i).label = 17;
%     elseif i<204
%     test2(i).label = 18;
%     elseif i<217
%     test2(i).label = 19;
%     elseif i<227
%     test2(i).label = 20;
%     elseif i<237
%     test2(i).label = 21;
%     elseif i<253
%     test2(i).label = 22;
%     elseif i<263
%     test2(i).label = 23;
%     elseif i<277
%     test2(i).label = 24;
%     elseif i<293
%     test2(i).label = 25;
    end
end

%% compute feature histogram for each test image
H_test = zeros(size(test2,2), N);
for i = 1:size(test2,2)
    des_each = single(test2(i).description);
        for j = 1:size(des_each,2)
        [~, k] = min(vl_alldist2(des_each(:,j), centers, 'l2')) ;%l2 is Euclidean metric
        H_test(i,k) = H_test(i,k) + 1;
        end
    H_test(i,:) = H_test(i,:)./size(des_each,2);
end

%for i = 1:609
%    indices = find(min(peak2peak(H_test)));
%    H_test(:,indices) = [];
%end


%% Classifiacation

H_dis = transpose(H_test);

for i = 1:size(test2,2)
    [~, m] = min(vl_alldist2(H_dis(:,i), transpose(H), 'l2'));
    test2(i).classification = m;
end

%% Result
acc = 0
for i = 1:size(test2,2)
    if test2(i).label == test2(i).classification
        acc = acc+1;
    end
end
acc = acc/num_test;

M = zeros(num_class,num_class);
for i = 1:size(test2,2)
    M(test2(i).label,test2(i).classification) =  M(test2(i).label,test2(i).classification)+1;
end
for i = 1:length(M)
    M(i,:) = M(i,:)/sum(M(i,:));
end

