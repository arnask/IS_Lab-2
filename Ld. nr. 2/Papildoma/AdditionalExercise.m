clc
clear
x1 = 0.1:1/22:1;
x2 = 0.25:1/26:1;
d = (1 + 0.6 * sin (2 * pi * x1 / 0.7) + 0.3 * sin (2 * pi * x2))/2;
% First input to hidden layer weights
w1_11 = randn(1);
w1_21 = randn(1);
w1_31 = randn(1);
w1_41 = randn(1);
% Second input to hidden layer weights
w1_12 = randn(1);
w1_22 = randn(1);
w1_32 = randn(1);
w1_42 = randn(1);
% Hidden layer to output weights
w2_1 = randn(1);
w2_2 = randn(1);
w2_3 = randn(1);
w2_4 = randn(1);
% Hidden layer biases
b1_1 = randn(1);
b1_2 = randn(1);
b1_3 = randn(1);
b1_4 = randn(1);
% Output bias 
b2_1 = randn(1);
lr = 0.02;

for index = 1 : 100000
    
% Forward Pass
    for i = 1 : length(x1)
        V1_1(i) = x1(i)*w1_11 + x2(i)*w1_12 + b1_1;
        V1_2(i) = x1(i)*w1_21 + x2(i)*w1_22 + b1_2;
        V1_3(i) = x1(i)*w1_31 + x2(i)*w1_32 + b1_3;
        V1_4(i) = x1(i)*w1_41 + x2(i)*w1_42 + b1_4;
        y1_1(i) = 1/(1 + exp(-V1_1(i)));
        y1_2(i) = 1/(1 + exp(-V1_2(i)));
        y1_3(i) = 1/(1 + exp(-V1_3(i)));
        y1_4(i) = 1/(1 + exp(-V1_4(i)));
        v(i) = y1_1(i) * w2_1 + y1_2(i) * w2_2 + y1_3(i) * w2_3 + y1_4(i) * w2_4 + b2_1;
        y(i) = 1/(1 + exp(-v(i)));
        e(i) = d(i) - y(i);

    end
    % Output layer weights updating
    for i = 1 : length(x1)
        delta2 = y(i) * (1 - y(i)) * e(i);
        w2_1 = w2_1 + lr * delta2 * y1_1(i);
        w2_2 = w2_2 + lr * delta2 * y1_2(i);
        w2_3 = w2_3 + lr * delta2 * y1_3(i);
        w2_4 = w2_4 + lr * delta2 * y1_4(i);
        b2_1 = b2_1 + lr * delta2;
    end
    
    % Hidden layer weights updating
    for i = 1 : length(x1)
        
        % Delta calculation
        delta1_1 = y1_1(i) * (1 - y1_1(i)) * e(i) * delta2 * w2_1;
        delta1_2 = y1_2(i) * (1 - y1_2(i)) * e(i) * delta2 * w2_2;
        delta1_3 = y1_3(i) * (1 - y1_3(i)) * e(i) * delta2 * w2_3;
        delta1_4 = y1_4(i) * (1 - y1_4(i)) * e(i) * delta2 * w2_4;
        
        % Weights for x1
        w1_11 = w1_11 + lr * delta1_1 * x1(i);
        w1_21 = w1_21 + lr * delta1_2 * x1(i);
        w1_31 = w1_31 + lr * delta1_3 * x1(i);
        w1_41 = w1_41 + lr * delta1_4 * x1(i);
        
        % Weights for x2
        w1_12 = w1_12 + lr * delta1_1 * x2(i);
        w1_22 = w1_22 + lr * delta1_2 * x2(i);
        w1_32 = w1_32 + lr * delta1_3 * x2(i);
        w1_42 = w1_42 + lr * delta1_4 * x2(i);
        
        % Biases updating
        b1_1 = b1_1 + lr * delta1_1;
        b1_2 = b1_2 + lr * delta1_2;
        b1_3 = b1_3 + lr * delta1_3;
        b1_4 = b1_4 + lr * delta1_4;
    end
end

% Passing forward again to check how network learned
for i = 1 : length(x1)
    V1_1(i) = x1(i)*w1_11 + x2(i)*w1_12 + b1_1;
    V1_2(i) = x1(i)*w1_21 + x2(i)*w1_22 + b1_2;
    V1_3(i) = x1(i)*w1_31 + x2(i)*w1_32 + b1_3;
    V1_4(i) = x1(i)*w1_41 + x2(i)*w1_42 + b1_4;
    y1_1(i) = 1/(1 + exp(-V1_1(i)));
    y1_2(i) = 1/(1 + exp(-V1_2(i)));
    y1_3(i) = 1/(1 + exp(-V1_3(i)));
    y1_4(i) = 1/(1 + exp(-V1_4(i)));
    v(i) = y1_1(i) * w2_1 + y1_2(i) * w2_2 + y1_3(i) * w2_3 + y1_4(i) * w2_4 + b2_1;
    y(i) = 1/(1 + exp(-v(i)));
    e(i) = d(i) - y(i);
end
plot(x1,y,"k*", x1,d,"gx")