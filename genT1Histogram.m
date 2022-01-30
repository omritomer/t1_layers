function T1Histogram = genT1Histogram(T1map,normM0map)
%Generate a whole-brain histogram of T1-values proportional to their
%partial volume
%
%   This function generates a whole-brain histogram of the T1-values that
%   is proportional to their partial volume, represented by each
%   T1-component's M0 value normalized per-voxel.
%
%
% Syntax:
%   T1Histogram = genT1Histogram(T1map,normM0map)
%
%
% Description:
%   T1Histogram = genT1Histogram(T1map,normM0map) returns a whole brain
%   histogram vector of the T1-values from T1map proportional to those
%   values' matching normalized M0 values (from normM0map).
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
%
% Output arguments:
%
%   T1Histogram - a row vector of the whole-brain T1-values ditribution
%   (from T1map) proportional to their partial volume (from normM0map).

pM0map=round(normM0map.*100,0); % rounded percentage matrix of each T1 value (in single voxel)
T1Histogram=[]; % initialize T1Histogram

T1vec=reshape(T1map,[1,size(T1map,1)*size(T1map,2)*size(T1map,3)*size(T1map,4)]); % reshape T1map as a linear vector
pM0vec=reshape(pM0map,[1,size(pM0map,1)*size(pM0map,2)*size(pM0map,3)*size(pM0map,4)]); % reshape pM0map as a linear vector

for ind=1:length(T1vec) % for every T1value         
    T1Histogram(end+1:end+pM0vec(ind)) = T1vec(ind); % insert the T1-value in the ind position into the next pM0vec(ind) elements
end

end
