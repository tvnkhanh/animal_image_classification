outputFolder = fullfile('animals');
rootFolder = fullfile(outputFolder, 'animals');

categories = {'wolf', 'whale', 'turkey', 'pig'};

imds = imageDatastore(fullfile(rootFolder, categories), "LabelSource", "foldernames");

tbl = countEachLabel(imds);
minSetCount = min(tbl{:, 2});

imds = splitEachLabel(imds, minSetCount, 'randomized');

wolf = find(imds.Labels == 'wolf', 1);
whale = find(imds.Labels == 'whale', 1);
turkey = find(imds.Labels == 'turkey', 1);
pig = find(imds.Labels == 'pig', 1);

% figure;
% subplot(2, 2, 1);
% imshow(readimage(imds, wolf));
% subplot(2, 2, 2);
% imshow(readimage(imds, whale));
% subplot(2, 2, 3);
% imshow(readimage(imds, turkey));
% subplot(2, 2, 4);
% imshow(readimage(imds, pig));

net = resnet50();
% figure;
% plot(net);
% title('Architecture of ResNet-50');
% set(gca, 'YLim', [150 170]);

net.Layers(1);
net.Layers(end);

numel(net.Layers(end).ClassNames); 
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomized');

imageSize = net.Layers(1).InputSize; 

augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, ...
    'ColorPreprocessing', 'gray2rgb');

augmentedTestSet = augmentedImageDatastore(imageSize, testSet, ...
    'ColorPreprocessing', 'gray2rgb');

w1 = net.Layers(2).Weights;
w1 = mat2gray(w1);

% figure;
% montage(w1);
% title('First Convolutional Layer Weight');

featureLayer = 'fc1000';
trainingFeatures = activations(net, ...
    augmentedTrainingSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

trainingLabels = trainingSet.Labels;
classifier = fitcecoc(trainingFeatures, trainingLabels, 'Learners','linear', ...
    'Coding','onevsall', 'ObservationsIn', 'columns');

testFeatures = activations(net, ...
    augmentedTestSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns'); 

predictLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

testLabels = testSet.Labels;
confMat = confusionmat(testLabels, predictLabels);
confMat = bsxfun(@rdivide, confMat, sum(confMat, 2));

mean(diag(confMat));

newImage = imread("test\Southern_right_whale.jpg");

ds = augmentedImageDatastore(imageSize, newImage, ...
    'ColorPreprocessing', 'gray2rgb');

imageFeatures = activations(net, ...
    ds, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

label = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');

sprintf('The loaded image belongs to %s class', label)
