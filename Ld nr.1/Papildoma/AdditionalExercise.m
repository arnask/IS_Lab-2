clear
clc
% Reading text file
[data_X1,data_X2,data_T] = textread("Data.txt", "%f,%f,%f");
data = [data_X1 data_X2 data_T]
training_data = data(1:2:end, :);
% Spliting data to classes
a = SplitToClasses(training_data)
mapKeys = keys(a);
mapValues = values(a);
% Calculating average, standard deviation for each column
% And separating columns by a class
summarized_columns = containers.Map('KeyType','double','ValueType','any');
for i = 1 : length(mapKeys)
    for j = 1: size(mapValues{i},2)-1
        if isKey(summarized_columns, mapKeys{i}) == 0
        summarized_columns(mapKeys{i}) = StandardDeviation(mapValues{i}(:,j))
        else
            summarized_columns(mapKeys{i}) = [summarized_columns(mapKeys{i}); StandardDeviation(mapValues{i}(:,j))]
        end
    end
end
% Predicting
prediction = CalculateClassProbabilities(summarized_columns,training_data, data(9,:))
predictionKeys = keys(prediction);
fprintf("{%i: %f, %i: %f]", predictionKeys{1}, prediction(predictionKeys{1}),predictionKeys{2}, prediction(predictionKeys{2}))

function [separated]  = SplitToClasses (training_data)
    separated = containers.Map('KeyType','double','ValueType','any');
    for i = 1 : length(training_data)
        vector =  training_data(i,:);
        class_value = vector(end);
        if isKey(separated, class_value) == 0
            separated(class_value) = vector;
        else
            separated(class_value) = [separated(class_value); vector];
        end
    end
end
% Average calculation
function [avg] = Average(data)
    avg = sum(data) / length(data);
end
% Standard deviation calculation
% And columns summary combination
function [summary] = StandardDeviation(data)
    avg = Average(data);
    variance = 0;
    for i = 1 : length(data)
    variance = variance + ((data(i) - avg) ^2 / (length(data)-1));
    end
    variance = sqrt(variance);
    summary = [avg variance length(data)];
end
% Calculating probabilities for each class
function [probabilities] = CalculateClassProbabilities(summarized_columns,training_data, prediction)
    numberOfTrainingRecords = length(training_data);
    probabilities = containers.Map('KeyType','double','ValueType','any');
    mapKeys = keys(summarized_columns);
    mapValues = values(summarized_columns);
    for i = 1 : length(mapKeys)
        probabilities(mapKeys{i}) = mapValues{i}(1,3) / numberOfTrainingRecords;
        for j = 1 : length(mapKeys)
            mean = mapValues{i}(j,1);
            stdev = mapValues{i}(j,2);
            count = mapValues{i}(1,3);
            probabilities(mapKeys{i}) = probabilities(mapKeys{i}) * calculate_probability(prediction(j), mean, stdev);
        end
    end
end
% Gausian probability density function
function [probability] = calculate_probability(x, mean, stdev)
      exponent = exp(-((x-mean)^2 / (2 * stdev^2 )));
      probability = (1 / (sqrt(2 * pi) * stdev)) * exponent;
end