% Two Class(Binary) Classification Demo
%
%
% This code implements basic SVM classification for 10 datasets
% taken from UC Irvine Machine Learning Repository
%
% Misclassification Rates are given at the end(command window)
%
% Datasets:
%   1 --> Breast Cancer Wisconsin Original
%   2 --> Climate Model Simulation Crashes
%   3 --> Credit Approval
%   4 --> Diabetic Retinopathy Debrecen
%   5 --> Echocardiogram Data
%   6 --> Hepatitis Domain
%   7 --> Ionosphere
%   8 --> Parkinsons Data Set
%   9 --> Pima Indians Diabetes
%   10 --> US Congressional Voting Records
%
%
%
%
% Matlab R2014a
% 30/10/2017 - Furkan Burak BAÐCI


clear all
close all
clc

% Adjust the directory!!!
% cd C:\..\classification_project\Matlab\Data\UCI

files = dir('*.mat');
for l=1:length(files)
    
    eval(['load ' files(l).name ]); % loading data set
    rng(1); % For reproducibility
    
    SVMModel = fitcsvm(X,Y,'Standardize',true,'KernelFunction','RBF',...
        'KernelScale','auto'); % Train SVM Classification model
    
    CVSVMModel = crossval(SVMModel); % Cross validation
    
    classLoss = kfoldLoss(CVSVMModel); % Misclassification
    
    X = ['Misclassification for Dataset   '...
        ,num2str(l),'   is   ',num2str(classLoss)];
    disp(X);
    
end