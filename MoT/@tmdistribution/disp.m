function disp(obj)
%DISP Display a Gaussian mixture distribution object.
%   DISP(OBJ) prints a text representation  of the gmdistribution
%   OBJ, without printing the object name.  In all other ways it's
%   the same as leaving the semicolon off an expression.
%
%   See also GMDISTRIBUTION, GMDISTRIBUTION/DISPLAY.

%   Copyright 2008-2011 The MathWorks, Inc.



isLoose = strcmp(get(0,'FormatSpacing'),'loose');

if (isLoose)
end



if obj.NComponents > 0 && obj.NDimensions < 6
    for j =1 :obj.NComponents
       
        disp(obj.mu(j,:));
        
    end
end

if (isLoose)
    fprintf('\n');
end
