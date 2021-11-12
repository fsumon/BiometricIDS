clear all;
Prepare training dataset
load(fullfile('ECGPPGNB.mat'));
dataMAT = ECGPPGNEW.Data;
[rowZ,col] = size(dataMAT);
% Taking 50% of row for training
hf = round(rowZ/2,0);

TrainTestData = dataMAT(1:hf,:);
[row,col] = size(TrainTestData);
TestData = dataMAT(hf+1:rowZ,:);

%sigLen = 500;  % Must be an even number, 10 features SVM:90.9% KNN:95.5%
sigLen = 1000;  % Must be an even number 95.2%, 96.8%
iMax = round(col/sigLen,0);  % Max number of samples of each user
gSize = 1;      % Auth group size

% Taking a small group of X users as authenticated participants.
for i = 1 : iMax-1
        % Only taking first gSize number of users data as AUTH users
    if i == 1
        FinalDataSegment = TrainTestData(1:gSize,1:sigLen);
    else 
        DataSegment = TrainTestData(1:gSize , i*sigLen+1 : i*sigLen+sigLen);
        FinalDataSegment = vertcat(FinalDataSegment,DataSegment);
    end

    for j = 1 : gSize
        train_labelA{(i-1) * gSize+j,1} = 'AUTH';
    end    
end


[rowA,colA] = size(train_labelA);
%sZ = round(row/gSize,0);
% We are using 30% users for training
sZ = round(rowA*0.3,0);
% Use only sZ number of train samples for AUTH user to match with rest of
% the dataset of NAUTH users
DataAuthTrain = FinalDataSegment(1:sZ,:);
trainData_1 = DataAuthTrain;
DataAuthTrainLabel = train_labelA(1:sZ);
train_label_1 = DataAuthTrainLabel;

% Repeating training data samples of AUTH user to enforce overtraining
for n = 1:5
    trainData_1 = vertcat(trainData_1,trainData_1);
    train_label_1 = vertcat(train_label_1,train_label_1);
end

DataAuthTest = FinalDataSegment(sZ+1:rowA,:);
testData_1 = DataAuthTest;
DataAuthTestLabel = train_labelA(sZ+1:rowA);
test_label_1 = DataAuthTestLabel;

% We need similar number of NAUTH data points
[rowB,colB] = size (trainData_1);
% We can only generate rowB number of NAUTH data points

k=1;
% Taking a large group of Y users as non-authenticated participants
%for i = gSize+1 : iMax-1
if gSize == 1
    p=1;
else
    p=gSize+1;
end
for i = p : iMax-1 
        % Skipping the first gSize number of users data that was used AUTH users
        % so next data will be used for NAUTH users
    if i == p
        FinalDataSegment = TrainTestData(p+1:row,1:sigLen);
    else 
        DataSegment = TrainTestData(p+1:row, k*sigLen+1 : k*sigLen+sigLen);
        FinalDataSegment = vertcat(FinalDataSegment,DataSegment);
    end
    
    count = 1;
    for j = 1 : row-gSize
        train_labelB{(k-1) * (row-gSize)+j,1} = 'NAUTH';
    end    
    %train_label_2{i,1} = 'NAUTH';
    k = k + 1;

%     [rowC,colC] = size (FinalDataSegment);
%     if rowC > sZ
%         break;
%     end 
end

[rowX,colX] = size(train_label_1);
[rowD,colD] = size(train_labelB);
trainData_2     = FinalDataSegment(1:rowX,:);
train_label_2   = train_labelB(1:rowX);
testData_2      = FinalDataSegment(rowX+1:rowD,:);
test_label_2    = train_labelB(rowX+1:rowD);


Prepare testing data [ From same dataset 50% was reserved]

%load(fullfile('NewPPGData.mat'));
%TestData = newPPGTest.Data;
%UsrNo = 120;
%test_label = ECG120.Labels;

[rowE,colE] = size (TestData);


for m = 1:round(colE/sigLen,0)
        % Only taking first gSize number of users data as AUTH users
        
    if m*sigLen+sigLen > colE
        break;
    end

    if m == 1
        testData_3 = TestData(1:rowE,1:sigLen);
    else 
        DataSegment = TestData(1:rowE , m*sigLen+1 : m*sigLen+sigLen);
        testData_3 = vertcat(testData_3,DataSegment);
    end
    
    for j = 1 : rowE
        test_label_3{(m-1) * rowE+j,1} = 'NAUTH';
    end    
end



Prepare testing data [ From new dataset all treated as NAUTH ]

% load(fullfile('ECG120.mat'));
% NewTestData = ECG120.Data;
% 
% [rowE,colE] = size (NewTestData);
% 
% for m = 1:round(colE/sigLen,0)
%         % Only taking first gSize number of users data as AUTH users
%         
%     if m*sigLen+sigLen > colE
%         break;
%     end
% 
%     if m == 1
%         testData_4 = NewTestData(1:rowE,1:sigLen);
%     else 
%         DataSegment = NewTestData(1:rowE , m*sigLen+1 : m*sigLen+sigLen);
%         testData_4 = vertcat(testData_4,DataSegment);
%     end
%     
%     for j = 1 : rowE
%         test_label_4{(m-1) * rowE+j,1} = 'NAUTH';
%     end    
% end

Contating all the training and testing sets
% Concating all the train and test data
trainData = vertcat(trainData_1,trainData_2);
train_label = vertcat(train_label_1,train_label_2);

% testData  = vertcat(testData_1,testData_2,testData_3,testData_4);
% test_label = vertcat(test_label_1,test_label_2,test_label_3,test_label_4);
testData  = vertcat(testData_1,testData_2,testData_3);
test_label = vertcat(test_label_1,test_label_2,test_label_3);
Feature extraction
%timeWindow = 8192;
timeWindow = sigLen; % for ECGPPG, GSR

AROrder = 12;
%AROrder = 16;
%transformLevel = 8;
transformLevel = 8;
%transformLevel = 2;
[trainFeatures,testFeatures,featureindices] = featureExtraction(trainData,testData,timeWindow,AROrder,transformLevel);

allFeatures = [trainFeatures;testFeatures];
allLabels = [train_label;test_label];


Feature Selection

   [idx,scores] = fscchi2(trainFeatures,train_label);
   bar(scores(idx))
    %idx=sort(idx);
    xlabel('Predictor rank')
    ylabel('Predictor importance score')
    
    selected_feature_indx =  idx(:,1:16); %85 good
    % Plot feature weights
    %stem(mdl.FeatureWeights,'bo');
    bar(scores(idx))
    xlabel('Predictor rank')
    ylabel('Predictor importance score')
    
    % save for future reference
    save('SelectedFeatures', 'selected_feature_indx');

SVM classification
rng(1)
template = templateSVM(...
    'KernelFunction','polynomial',...
   'KernelScale','auto',...
    'BoxConstraint',1,...
    'Standardize',true);
% testing with rbf

model = fitcecoc(...
     trainFeatures(:,selected_feature_indx),...
     train_label,...
     'Learners',template,...
     'Coding','onevsone', ...
     'ClassNames',{'AUTH','NAUTH'});


predLabels = predict(model,testFeatures(:,selected_feature_indx));

[label,score]=predict(model,testFeatures(:,selected_feature_indx));
classOrder = model.ClassNames;
kfoldmodel = crossval(model,'KFold',10);
classLabels = kfoldPredict(kfoldmodel);
loss = kfoldLoss(kfoldmodel)
correctPredictions = strcmp(predLabels,test_label);
%correctPredictions = strcmp(predLabels,train_label);
[confusionMatrix] = confusionmat(test_label,predLabels);
plotconfusion(categorical(test_label),categorical(predLabels));
%mdlSVM = fitcsvm(testFeatures,predLabels,'Standardize',true);
%mdlSVM = fitPosterior(mdlSVM);


calculate prediction results
truePositive = confusionMatrix(1,1);
falsePositive = sum(confusionMatrix(:,1)) - confusionMatrix(1,1);
falseNegative = sum(confusionMatrix(1,:)) - confusionMatrix(1,1);
trueNegative = confusionMatrix(2,2);

precision = truePositive / (truePositive + falsePositive);
recall = truePositive / (truePositive + falseNegative);
trueNegativeRate = trueNegative / (trueNegative+falsePositive);
falseNegativeRate = falseNegative / (falseNegative+truePositive);
falsePositiveRate= 1 - trueNegativeRate;
truePositiveRate= truePositive/(truePositive + falseNegative);
f1 = 2 * precision * recall / (precision + recall);
EER=(falsePositive+falseNegative) / (truePositive+falsePositive+falsePositive+trueNegative);

display result
%testAccuracy = sum(correctPredictions) / length(testLabel)*100
testAccuracy = sum(correctPredictions) / length(test_label)*100

confusionMatrixTable = array2table([truePositive falsePositive;...
    falseNegative trueNegative],'VariableNames',{'Positive','Negative'},'RowNames',...
    {'Positive','Negative'})

resultTable = array2table([precision recall f1 trueNegativeRate ...
    falseNegativeRate falsePositiveRate],'VariableNames',{'Precision','Recall','F1_Score',...
    'True Negative Rate', 'False Negative Rate', 'False Positive Rate' },'RowNames',...
    {'value',});

disp(confusionMatrixTable)
disp(resultTable);


KNN  Algorithm 
model=fitcgam(trainFeatures(:,selected_feature_indx),train_label);  
predLabels = predict(model,testFeatures(:,selected_feature_indx));

[label,score]=predict(model,testFeatures(:,selected_feature_indx));
classOrder = model.ClassNames;
kfoldmodel = crossval(model,'KFold',10);
classLabels = kfoldPredict(kfoldmodel);
loss = kfoldLoss(kfoldmodel)
%correctPredictions = strcmp(predLabels,testLabel);
correctPredictions = strcmp(predLabels,test_label);
[confusionMatrix] = confusionmat(test_label,predLabels);
plotconfusion(categorical(test_label),categorical(predLabels));
%mdlSVM = fitcsvm(testFeatures,predLabels,'Standardize',true);
%mdlSVM = fitPosterior(mdlSVM);


calculate prediction results
truePositive = confusionMatrix(1,1);
falsePositive = sum(confusionMatrix(:,1)) - confusionMatrix(1,1);
falseNegative = sum(confusionMatrix(1,:)) - confusionMatrix(1,1);
trueNegative = confusionMatrix(2,2);

precision = truePositive / (truePositive + falsePositive);
recall = truePositive / (truePositive + falseNegative);
trueNegativeRate = trueNegative / (trueNegative+falsePositive);
falseNegativeRate = falseNegative / (falseNegative+truePositive);
falsePositiveRate= 1 - trueNegativeRate;
truePositiveRate= truePositive/(truePositive + falseNegative);
f1 = 2 * precision * recall / (precision + recall)
EER=(falsePositive+falseNegative) / (truePositive+falsePositive+falsePositive+trueNegative)

display result
%testAccuracy = sum(correctPredictions) / length(testLabel)*100
testAccuracy = sum(correctPredictions) / length(test_label) * 100

confusionMatrixTable = array2table([truePositive falsePositive;...
    falseNegative trueNegative],'VariableNames',{'Positive','Negative'},'RowNames',...
    {'Positive','Negative'});

resultTable = array2table([precision recall f1 trueNegativeRate ...
    falseNegativeRate falsePositiveRate],'VariableNames',{'Precision','Recall','F1_Score',...
    'True Negative Rate', 'False Negative Rate', 'False Positive Rate' },'RowNames',...
    {'value',});

disp(confusionMatrixTable)
disp(resultTable)

