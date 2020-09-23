clear
% Reading text file
[data_X1,data_X2,data_T] = textread("Data.txt", "%f,%f,%f");
X1 = data_X1(1:2:end);
X2 = data_X2(1:2:end);
T = data_T(1:2:end);
% Defining random values
W1 = randn(1)
W2 = randn(1)
B = randn(1)
% Learning rate
lr = 0.01;
for i = 1 : 10000
    [e,y] = ErrorCalculation(X1,X2,W1,W2,B,T);
    [W1,W2,B] = ParametersUpdator(W1,W2,B,X1,X2,lr,e);
end
% Testing on full data set
[e,y] = ErrorCalculation(data_X1,data_X2,W1,W2,B,data_T)
