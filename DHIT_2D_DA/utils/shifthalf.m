function res = shifthalf(input, dir)
    res = 0.5*(circshift(input,-dir) + input);
end