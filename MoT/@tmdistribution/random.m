function [y, compIdx] = random(obj,n)
%RANDOM Random vector generation. 
%   Y = RANDOM(OBJ) generates an random row vector Y drawn from the
%    Gaussian mixture distribution with parameters given by OBJ.
%
%   Y = RANDOM(OBJ,N) generates an N-by-D matrix Y. Each row of Y is a
%   random vector drawn from the Gaussian mixture distribution with
%   parameters given by OBJ.
%
%   [Y, COMPIDX] = RANDOM(OBJ,N) returns an N-by-1 vector COMPIDX
%   which contains the index of the component used to generate each row of
%   Y.

%   Copyright 2007 The MathWorks, Inc.


if nargin < 2 || isempty(n)
    n = 1;
end

[y, compIdx] = tmrnd(obj.mu,obj.Sigma,obj.phi,obj.nu,n);

end
