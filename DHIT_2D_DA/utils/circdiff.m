function res = circdiff(input, dir)
    res = circshift(input,-dir) - input;
end