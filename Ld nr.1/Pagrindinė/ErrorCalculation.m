function [e,y] = ErrorCalculation(X1, X2, W1, W2, B, T)
 e = zeros(1,length(X1));
    for i = 1 : length(X1)
        % Calculating V
        v = X1(i) * W1 + X2(i) * W2 + B;
        if v > 0
            y(i) = 1;
        else 
            y(i) = -1;
        end
        e(i) =(T(i) - y(i));
    end
end

