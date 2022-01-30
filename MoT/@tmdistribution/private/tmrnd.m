function [y,compIdx] = tmrnd(mu,sigma,phi,nu,n)
%GMRND Random vectors from a multivariate Gaussian mixture model.
%   Y = GMRND(MU,SIGMA,P,N) returns an N-by-D matrix Y of random vectors
%   chosen from the D-dimensional Gaussian mixture model whose K components
%   are each multivariate normal distributions with mean vectors given by MU,
%   and covariance matrices given by SIGMA.  MU is a K-by-D matrix, where
%   MU(J,:) is the mean of component J.  SIGMA is a D-by-D-by-K array
%   containing K symmetric, positive semi-definite matrices, where
%   SIGMA(:,:,J) is the covariance of component J.  SIGMA can also be a D-by-D
%   matrix, in such case all the components use the same covariance.  P is a
%   1-by-K vector of mixture probabilities, where P(J) contains the
%   probability of component J.  If P does not sum to 1, GMRND normalizes
%   it.  If P is not given, each component will get equal probability.  The
%   default value for N is one.
%
%   [Y, COMPIDX] = GMRND(MU,SIGMA,P,N) returns an N-by-1 vector COMPIDX
%   which contains the index of the component used to generate each row of
%   Y.
%
%   Example:
%
%      mu = [0 0; -8 -8];
%      sigma = cat(3, [2 0; 0 .5], [1 0; 0 1]);
%      [y,compIdx] = gmrnd(mu,sigma,[0.4,0.6],10000);
%      hist3(y,[25,25]);
%
%   See also MVNRND, MVNPDF, MVNCDF, NORMRND.

%   Y = GMRND(MU,SIGMA,P,N,T) provides the Cholesky factors T of SIGMA, so
%   that SIGMA(:,:,J) == T(:,:,J)'*T(:,:,J) if SIGMA is a 3D array or SIGMA
%   == T'*T if SIGMA is a matrix.  No error checking is done on T except
%   checking the size of T.
%
%   [Y,T] = GMRND(...) returns the Cholesky factors T, so they can be
%   re-used to make later calls more efficient.

%   Copyright 2007 The MathWorks, Inc.




if nargin < 2 || isempty(mu) || isempty(sigma)
    error(message('stats:gmdistribution:gmrnd:TooFewInputs'));
elseif ndims(mu) ~= 2
    error(message('stats:gmdistribution:gmrnd:BadMu'));
elseif ndims(sigma) ~= 2
    error(message('stats:gmdistribution:gmrnd:BadSigma'));
end
[K,d] = size(mu);


if nargin < 3  || isempty(phi)
    phi = repmat(1/K,[1,K]); % default equal component probability
elseif ~isvector(phi)
    error(message('stats:gmdistribution:gmrnd:BadP'));
elseif length(phi)~=K
    error(message('stats:gmdistribution:gmrnd:MuPSizeMismatch'));
elseif any(phi<0 | phi >1) 
    error(message('stats:gmdistribution:gmrnd:InvalidP'));
end

if nargin < 4 || isempty(n)
    n = 1;
elseif ~isnumeric(n) ||~isscalar(n) ||n<=0 ||n ~= round(n)
    error(message('stats:gmdistribution:gmrnd:BadN'));
end

% Randomly pick from the components.
compIdx = randsample(length(phi),n,true,phi/sum(phi));
y = zeros(n,1);



        for i = 1:K
            mbrs = find(compIdx == i);
            y(mbrs) = random('tlocationscale',mu(i,:),sqrt(sigma(i,:)),nu(i,:),size(mbrs));
        end
end




