# Face-Emotion-Recognition-on-FER2013-Dataset
%% =================== FER2013 Pipeline (EfficientNet-B0 Fine-Tuned) ===================
%% ------------------- Paths -------------------
trainFolder = "D:\Samra\PaperFinal\FER2013\train";
testFolder  = "D:\Samra\PaperFinal\FER2013\test";
trainImds = imageDatastore(trainFolder, ...
  'IncludeSubfolders',true, ...
  'LabelSource','foldernames');
testImds  = imageDatastore(testFolder, ...
  'IncludeSubfolders',true, ...
  'LabelSource','foldernames');
% Split training: 80% train / 20% validation
[trainImds, valImds] = splitEachLabel(trainImds, 0.8, 'randomized');
%% ------------------- Detectors -------------------
faceDetector = vision.CascadeObjectDetector();
eyeDetector  = vision.CascadeObjectDetector('EyePairBig');
%% ------------------- Preprocessing Function -------------------
function Iout = preprocessFERAligned(img, faceDetector, eyeDetector)
   if size(img,3)==3
       imgGray = rgb2gray(img);
   else
       imgGray = img;
   end
   bboxFace = step(faceDetector,imgGray);
   if isempty(bboxFace)
       imgFace = imgGray;
   else
       [~,idx] = max(bboxFace(:,3).*bboxFace(:,4));
       imgFace = imcrop(imgGray, bboxFace(idx,:));
   end
   bboxEyes = step(eyeDetector,imgFace);
   if ~isempty(bboxEyes)
       angle = 0;
       tform = affine2d([cosd(angle) sind(angle) 0; -sind(angle) cosd(angle) 0; 0 0 1]);
       imgFace = imwarp(imgFace, tform, 'OutputView', imref2d(size(imgFace)));
   end
   imgFace = imresize(imgFace, [224 224]);
   imgFace = adapthisteq(imgFace);
   imgFace = imgaussfilt(imgFace, 0.5);
   % Convert grayscale → RGB (EfficientNet requires 3 channels)
   Iout = repmat(imgFace,[1 1 3]);
end
%% ------------------- Preprocess Training Images + Oversample -------------------
tbl = countEachLabel(trainImds);
maxCount = max(tbl.Count);
preprocTrainFolder = fullfile(tempdir,'FER2013_Train');
if ~exist(preprocTrainFolder,'dir'), mkdir(preprocTrainFolder); end
balancedFiles = strings(0,1);
balancedLabels = categorical();
for i = 1:height(tbl)
   lbl = tbl.Label(i);
   files = trainImds.Files(trainImds.Labels == lbl);
   nRep = ceil(maxCount/numel(files));
   filesRep = repmat(files, nRep, 1);
   for j = 1:maxCount
       img = imread(filesRep{j});
       imgProcessed = preprocessFERAligned(img, faceDetector, eyeDetector);
       savePath = fullfile(preprocTrainFolder, sprintf('%s_%d.png', char(lbl), j));
       imwrite(imgProcessed, savePath);
       balancedFiles = [balancedFiles; savePath];
       balancedLabels = [balancedLabels; lbl];
   end
end
trainImdsBalanced = imageDatastore(balancedFiles);
trainImdsBalanced.Labels = balancedLabels;
%% ------------------- Preprocess Validation Images -------------------
preprocValFolder = fullfile(tempdir,'FER2013_Val');
if ~exist(preprocValFolder,'dir'), mkdir(preprocValFolder); end
valFiles = valImds.Files;
valLabels = valImds.Labels;
preprocessedValFiles = strings(0,1);
for i = 1:numel(valFiles)
   img = imread(valFiles{i});
   imgProcessed = preprocessFERAligned(img, faceDetector, eyeDetector);
   savePath = fullfile(preprocValFolder, sprintf('%s_%d.png', char(valLabels(i)), i));
   imwrite(imgProcessed, savePath);
   preprocessedValFiles = [preprocessedValFiles; savePath];
end
valImdsProcessed = imageDatastore(preprocessedValFiles);
valImdsProcessed.Labels = valLabels;
%% ------------------- Heavy Augmentation -------------------
augmenter = imageDataAugmenter( ...
   'RandRotation',[-25 25], ...
   'RandXTranslation',[-5 5], ...
   'RandYTranslation',[-5 5], ...
   'RandXReflection',true, ...
   'RandScale',[0.85 1.15]);
inputSize = [224 224];
augTrain = augmentedImageDatastore(inputSize, trainImdsBalanced, ...
  'DataAugmentation',augmenter, ...
  'OutputSizeMode','resize', ...
  'ColorPreprocessing','none');
augVal = augmentedImageDatastore(inputSize, valImdsProcessed, ...
  'OutputSizeMode','resize', ...
  'ColorPreprocessing','none');
%% ------------------- Load EfficientNet-B0 -------------------
net = efficientnetb0();   % <<--- Your new model
lgraph = layerGraph(net);
numClasses = numel(categories(trainImds.Labels));
%% ------------------- Replace Final Layers (Fine-Tuning) -------------------
newLearnRate = 10;
fcLayerName = 'efficientnet-b0|model|head|dense|MatMul';
newFC = fullyConnectedLayer(numClasses, ...
   "Name","efficient_fc", ...
   "WeightLearnRateFactor", newLearnRate, ...
   "BiasLearnRateFactor", newLearnRate);
newSoftmax = softmaxLayer("Name","Softmax");
newClassLayer = classificationLayer("Name","classification");
% Replace final layers in EfficientNet-B0
lgraph = replaceLayer(lgraph,"efficientnet-b0|model|head|dense|MatMul", newFC);
lgraph = replaceLayer(lgraph,"Softmax", newSoftmax);
lgraph = replaceLayer(lgraph,"classification", newClassLayer);
%% ------------------- Training Options -------------------
options = trainingOptions('adam', ...
   'MiniBatchSize', 32, ...
   'MaxEpochs', 30, ...
   'InitialLearnRate', 1e-4, ...
   'Shuffle','every-epoch', ...
   'ValidationData', augVal, ...
   'ValidationFrequency', 50, ...
   'LearnRateSchedule','piecewise', ...
   'LearnRateDropFactor',0.5, ...
   'LearnRateDropPeriod',5, ...
   'Verbose',true, ...
   'Plots','training-progress', ...
   'ExecutionEnvironment','gpu');
%% ------------------- Train Network -------------------
netTrained = trainNetwork(augTrain, lgraph, options);
%% ------------------- Save Trained Network -------------------
save('FER2013_EfficientNetB0_Preprocessed.mat','netTrained');


