
data_path='C:/Users/user/Desktop/matlab/Assignment06_data/Assignment06_data_reduced'; 

imds = imageDatastore([data_path,'/TrainingDataset/'], 'IncludeSubfolders', true, 'labelsource', 'foldernames')
bag = bagOfFeatures(imds);
classifier = trainImageCategoryClassifier(imds,bag);

imds = imageDatastore([data_path,'/TestDataset/'], 'IncludeSubfolders', true, 'labelsource', 'foldernames')
evaluate(classifier,imds)