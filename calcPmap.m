function [Pmap,S] = calcPmap(T1map,normM0map,TMModel)
%Generate a volume probability map for multiple component distributions
%extracted from whole-brain T1-values
%
%   This function calculates the volume probability in each voxel for
%   multiple component distributions of different brain tissue. These
%   distributions are extractedby fitting a Gaussian mixture model to the
%   whole-brain histogram of T1-values with the function fitGMModel.
%
%
% Syntax:
%   Pmap = calcPmap(T1map,normM0map,GMModel)
%   Pmap = calcPmap(T1map,normM0map,GMModel,Name,Value)
%   [Pmap,S] = calcPmap(______)
%
%
% Description:
%   Pmap = calcPmap(T1map,normM0map,GMModel) returns a probability map
%   (Pmap) of tissue volume of each voxel in T1map. The first three
%   dimensions of Pmap are the size of the first three dimensions of T1map.
%   The size ofthe fourth dimension is the size of the number of components
%   in the Gaussian mixture model distribution (GMModel).

%
%   [Pmap,S] = calcPmap(______) also returns a struct (S) with the various
%   variables and parameters calculated by the different steps of the
%   function.
%
%
% Input arguments:
%
%   T1map - an X-by-Y-by-Z-by-N matrix of brain T1-values. X, Y and Z are
%   the spatial coordinations, and N is the number of T1-components per
%   voxel. Outputed by: calcT1map.
%   normM0map - a matrix the same size of T1map of the initial 
%   magnetization values for each T1 component, normalized per voxel,
%   resulting in a partial volume map of the T1 components in T1map, so
%   that sum(normM0map(i,j,k,:)) == 1. Outputed by: calcT1map.
%   GMModel - a GMModel object with G components calculated from a 
%   proportional histogram of the values in T1map. Outputed by: fitGMModel.
%
% Output arguments:
%
%   Pmap - a X-by-Y-by-X-by-G matrix of the tissue probability of each
%   component K (from 1 to G) in every voxel. For every i,j,k, if
%   sum(T1map(i,j,k,:)) > 0, then sum(Pmap(i,j,k,:)) == 1.
%   S - a struct containing the following variables and parameters as
%   fields:
%          pK - p(k), the probability of each Gaussian component.
%          pT1 - p(T1), the probability density for each T1-value in T1map
%          given the whole-brain Gaussian mixture distribution.
%          PT1GivenK - p(T1|K), the probability density for each T1-value
%          in T1map given a specific tissue distribution K
%          PKGivenT1 - p(K|T1) the probability for a tissue distribution K
%          given a specific T1-value, calculated using Bayes' rule:
%          [p(K|T1) = (p(T1|K)*p(K))/p(T1)].

K = TMModel.NumComponents; % get number of tissue components from GMModel
[X,Y,Z,N] = size(T1map); % get size of T1map. X,Y,Z are spatial coordinates, N is number of T1 components

%Variable definitions:
% pK - The probability of each tissue component [equivalent to its
% proportion in the Gaussian mixture model]

% pT1 - The probability for a specific T1 value in the entire distribution
% [drawn from the entire Gaussian mixture model; pdf(GMM,T1)]

% pT1GivenK - The probability for a specific T1 value in a specific tissue
% distribution [drawn from a normal distribution with the mean and standard
% deviation of the tissue component;
% pdf('normal',T1,GMM.mu(K),sqrt(GMM.Sigma(K)))]

% pKGivenT1 - The probability for a specific K distribution given a
% specific T1 [calculated according to Bayes' rule as:
% pKGivenT1 = (pT1GivenK * pK) / pT1

pK = TMModel.phi; % Get pK (component proportion) from GMM
pT1 = zeros(size(T1map)); % intialize pT1 matrix
pT1GivenK = zeros([X Y Z N K]); % intialize pT1GivenK matrix
pKGivenT1 = zeros([X Y Z N K]); % intialize pKGivenT1 matrix

pT1(T1map>0) = pdf(TMModel,T1map(T1map>0)); % Calculate pT1  of all T1 values in T1map
T1mask = cat(5,T1map>0,false(X,Y,Z,N,K-1)); % Initialize mask to be used for calculation of each PT1GivenK, masking the first position in the 5th dimension (the dimension of K)
% Calculate pT1GivenK and pKGivenT1
for K = 1:K
    pT1GivenK(T1mask) = pdf('tlocationscale',T1map(T1map>0),TMModel.mu(K),sqrt(TMModel.Sigma(K)),TMModel.nu(K)); % Calculate pT1GivenK for the K position in the 5th dimension
    pKGivenT1(:,:,:,:,K) = pK(K) .* pT1GivenK(:,:,:,:,K) ./ pT1; % Calculate pKGivenT1 by using Bayes' rule
    T1mask = circshift(T1mask,1,5); % shift the mask to the next position (K+1) in the 5th dimension
end
pKGivenT1(isnan(pKGivenT1)) = 0; % Replace NaNs caused by zero division with zeros

% For every tissue component, normalize each KGivenT1 component by
% multiplying it by its normalized M0 value, and then sum them for each
% layer. In every non-zero voxel, the sum of probabilities across all the
%  tissue components will be zero [sum(Pmap(i,j,k,:)) == 0]:
Pmap = squeeze(sum(repmat(normM0map,[1 1 1 1 K]) .* pKGivenT1,4));

if nargout > 1 % Create struct S
    S = struct('pK',pK,'pT1',pT1,'pT1GivenK',pT1GivenK,'pKGivenT1',pKGivenT1);
end

% save([SubjectID '_GMM.mat','Pmap','pKGivenT1','pT1GivenK','pK','pT1','GMM','M0norm','params']);

end