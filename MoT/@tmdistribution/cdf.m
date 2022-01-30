function y=cdf(obj,X)
%CDF CDF for the Gaussian mixture distribution.
%   Y=CDF(OBJ,X) returns Y, a vector of length N containing the values of
%   the cumulative distribution function (CDF) for the gmdistribution OBJ,
%   evaluated at the N-by-D data matrix X. Rows of X correspond to points,
%   columns correspond to variables. Y(I) is the cdf value of point I.
%
%   See also GMDISTRIBUTION, GMDISTRIBUTION/PDF.

%   Copyright 2007 The MathWorks, Inc.


% Check for valid input


y = zeros(size(X,1),1);

for j=1:obj.NComponents
    c = normcdf(X(:,1),obj.mu(j),sqrt(obj.Sigma(j)));
    
    y = y + obj.phi(j) * c;
end
 
end

