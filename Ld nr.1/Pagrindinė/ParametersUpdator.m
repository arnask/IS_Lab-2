function [W1, W2, B] = ParametersUpdator(W1, W2, B, X1, X2, lr, e)
    for i = 1 : length(X1)
    W1 = W1 + lr * e(i) * X1(i);
    W2 = W2 + lr * e(i) * X2(i);
    B = B + lr * e(i);
    end
end

