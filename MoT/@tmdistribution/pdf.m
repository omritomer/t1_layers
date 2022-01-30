function y=pdf(obj,X)
% PDF PDF for a Gaussian mixture distribution.
%    Y = PDF(OBJ,X) returns Y, a vector of length N containing the
%    probability density function (PDF) for the gmdistribution OBJ,
%    evaluated at the N-by-D data matrix X. Rows of X correspond to points,
%    columns correspond to variables. Y(I) is the PDF value of point I.
%
%    See also GMDISTRIBUTION, GMDISTRIBUTION/CDF.

%   Copyright 2007 The MathWorks, Inc.


% Check for valid input

% narginchk(2,2);
% checkdata(X,obj);
mu = obj.mu;
Sigma = obj.Sigma;
phi = obj.phi;
nu = obj.nu;
N = length(X);
K = length(mu);
Lk = zeros(N,K);

for k=1:K
    Lk(:,k) = phi(k) .* ...
        ((gamma((nu(k)+1)/2))./(gamma(nu(k)/2)*sqrt(pi*nu(k)*Sigma(k)))) .* ...
        ((1 + (1/nu(k)) .* (((X-mu(k)).^2)./Sigma(k))).^(-(nu(k)+1)/2));
end
y = sum(Lk,2);
end