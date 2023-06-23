function [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
[trainedModel, validationRMSE] = trainRegressionModel(trainingData)
inputTable = trainingData;
predictorNames = {'pH', 'Concenctrationngml', 'VarName4', 'IpMBuA', 'EpV', 'ApuVA', 'IpMBuA1', 'EpV1', 'ApuVA1', 'IuA'};
predictors = inputTable(:, predictorNames);
response = inputTable.VarName12;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];

template = templateTree(...
    'MinLeafSize', 1, ...
    'NumVariablesToSample', 10);
regressionEnsemble = fitrensemble(...
    predictors, ...
    response, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', 19, ...
    'Learners', template, ...
    'LearnRate', 0.2376760912822294);

predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(regressionEnsemble, x);
trainedModel.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

trainedModel.RequiredVariables = {'ApuVA', 'ApuVA1', 'Concenctrationngml', 'EpV', 'EpV1', 'IpMBuA', 'IpMBuA1', 'IuA', 'VarName4', 'pH'};
trainedModel.RegressionEnsemble = regressionEnsemble;
% Predictor and response variable extraction
% This code processes the data into a form suitable for training a model.
inputTable = trainingData;
predictorNames = {'pH', 'Concenctrationngml', 'VarName4', 'IpMBuA', 'EpV', 'ApuVA', 'IpMBuA1', 'EpV1', 'ApuVA1', 'IuA'};
predictors = inputTable(:, predictorNames);
response = inputTable.VarName12;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];

function [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
inputTable = trainingData;
predictorNames = {'pH', 'Concenctrationngml', 'VarName4', 'IpMBuA', 'EpV', 'ApuVA', 'IpMBuA1', 'EpV1', 'ApuVA1', 'IuA'};
predictors = inputTable(:, predictorNames);
response = inputTable.VarName12;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];

template = templateTree(...
    'MinLeafSize', 6, ...
    'NumVariablesToSample', 8);
regressionEnsemble = fitrensemble(...
    predictors, ...
    response, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', 1000, ...
    'Learners', template, ...
    'LearnRate', 0.01);

predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(regressionEnsemble, x);
trainedModel.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

trainedModel.RequiredVariables = {'ApuVA', 'ApuVA1', 'Concenctrationngml', 'EpV', 'EpV1', 'IpMBuA', 'IpMBuA1', 'IuA', 'VarName4', 'pH'};
trainedModel.RegressionEnsemble = regressionEnsemble;

inputTable = trainingData;
predictorNames = {'pH', 'Concenctrationngml', 'VarName4', 'IpMBuA', 'EpV', 'ApuVA', 'IpMBuA1', 'EpV1', 'ApuVA1', 'IuA'};
predictors = inputTable(:, predictorNames);
response = inputTable.VarName12;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedModel.RegressionEnsemble, 'KFold', 50);

% Calculate validation predictions
validationPredictions = kfoldPredict(partitionedModel);

% Verification RMSE Calculation
validationRMSE = sqrt(kfoldLoss(partitionedModel, 'LossFun', 'mse'));

regressionLearner