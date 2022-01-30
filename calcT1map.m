function [T1map,M0map,normM0map,Nmap,Errmap,MSEmap,PredictedMmap,IRmask] = calcT1map(IRdata,TIlist,varargin)
%Calculate a whole-brain map of T1-values in sub-voxel resolution
%
%   This function calculates the T1-values for a whole-brain inversion
%   recovery (IR) data set. The function is given a data set, a list of the
%   inversion times (TI) and the maximum number of components per voxel,
%   and outputs a 4-dimensional T1map (three dimensions are spatial, the
%   fourth dimension is the sub-voxel dimension), a matching map of M0
%   values, and also a map of those M0 values normalized per voxel.
%
%
% Syntax:
%   [T1map,M0map,normM0map] = calcT1map(IRdata,TIlist)
%   [T1map,M0map,normM0map,Nmap] = calcT1map(IRdata,TIlist)
%   [_______] = calcT1map(IRdata,TIlist,Name,Value)
%
%
% Description:
%   [T1map,M0map,normM0map] = calcT1map(IRdata,TIlist,Nmax) returns a map
%   of sub-voxel T1 values (T1map) fitted to the data(IRdata and TIlist)
%   calculated for each voxel. The function also outputs a map of M0 values
%   corresponding to the T1 values (M0map) and a map of M0 values
%   normalized in each voxel (normM0map) [such that:
%   sum(normM0map(i,j,k,:)) == 1].
%   [T1map,M0map,normM0map,Nmap] = also outputs a map the size of the the
%   first three dimensions of T1map, containing the number of sub-voxel
%   components for each voxel
%   [_______] = calcT1map(______,Name,Value) also supplies optimization
%   parameters (see below) and threshold parameter.
%
%
% Input arguments:
%
%   IRdata - an X-by-Y-by-Z-by-IR data set of whole brain IR data set, so
%   that for every D from 1 to IR, IRdata(:,:,:,D) represent an IR scan
%   with a different TI.
%   TIlist - a vector with IR elements corresponding to the inversion times
%   of IRdata.
%   Name-Value pairings:
%          Nmax - a scalar representing the maximum number of components in
%          each voxel to fit to the data.
%          TolFun - see calcT1
%          TolX - see calcT1
%          DiffMaxChange - see calcT1
%          DiffMinChange - see calcT1
%          MaskingThreshold - threshold for masking the data. Voxel's whose
%          mean signal is lower than MaskingThreshold are treated as
%          no signal and therefore no T1-components are calculated for
%          them.
%
% Output arguments:
%
%   T1map - an X-by-Y-by-Z-by-max(Nmap(:)) matrix of the whole-brain
%   T1-values in sub-voxel resolution
%   M0map - an X-by-Y-by-Z-by-max(Nmap(:)) matrix of the initial
%   magnetizationvalues matching to the T1 values in T1map.
%   normM0 - an X-by-Y-by-Z-by-max(Nmap(:)) matrix of the normalized values
%   of M0map, so that sum(normM0map(i,j,k,:)) == 1.
%   Nmap - an X-by-Y-by-Z map of the number of components calculated for
%   each voxel

% Set default optimization parameters for lsqnonlin
Nmax = floor(size(IRdata,4)/4); % default maximum number of components per voxel
TolFun = 0.1;
TolX = 0.1;
DiffMaxChange = 0.1;
DiffMinChange = 0.0001;
% Set default threshold parameter:
MaskingThreshold = 100; % Threshold for mean signal of voxel
maskCreated = false;

% Get Name-Value pairs of parameters.
for ind=1:2:length(varargin)
    switch(lower(varargin{ind}))
        case 'nmax'
            Nmax = varargin{ind+1};
            if 2*Nmax > size(IRdata,4)
                error('More parameters than equations');
            end
        case 'tolfun'
            TolFun = varargin{ind+1};
        case 'tolx'
            TolX = varargin{ind+1};
        case 'diffmaxchange'
            DiffMaxChange = varargin{ind+1};
        case 'diffminchange'
            DiffMinChange = varargin{ind+1};
        case 'maskingthreshold'
            MaskingThreshold = varargin{ind+1};
        case 'mask'
            mask = varargin{ind+1};
            maskCreated = true;
        otherwise
            error(['The parameter ' varargin{ind+1} ' is not recognized by makePmap.']);
            
    end
end


    [X,Y,Z,IR] = size(IRdata); % get size of data
% Initialize variables. Linear indexing is used instead of spatial indexing
% to enable parallel computing:
    T1map = zeros(X*Y*Z,Nmax); % initialize T1map
    M0map = zeros(X*Y*Z,Nmax); % initialize M0map
    normM0map = zeros(X*Y*Z,Nmax); % initialize normM0map
    Nmap = zeros(1,X*Y*Z); % initialize Nmap
    Errmap = zeros(1,X*Y*Z); % initialize Nmap
    MSEmap = zeros(1,X*Y*Z); % initialize Nmap
    PredictedMmap = zeros(X*Y*Z,IR);
    
% Reshape IRdata's spatial dimensions to a linear dimension to enable
% parallel computing:
    IR_lin = reshape(IRdata,[X*Y*Z,IR]);
 
% Create a mask of the data, where voxels with low signal across all
% inversion times are ignored, so as to not perform calculations on voxels
% outside the brain:

if maskCreated
    mask = reshape(mask,size(IR_lin,1),1);
else
    totIR = max(IR_lin,[],2); % get mean of each voxel
    totIR(totIR<MaskingThreshold) = 0; % zero voxel with mean lower than MaskingThreshold
    mask = false(size(IR_lin,1),1); % generate mask variable
    mask(totIR>0) = true; % create mask
end
%calculating T1 values using calcT1.   
    parfor l = 1:(X*Y*Z)
        if mask(l) % if mask == 1, perform optimization
            [T1,M0,normM0,~,Err,N,PredictedM,MSE] = calcT1(IR_lin(l,:), TIlist, Nmax,...
                'TolFun',TolFun,'TolX',TolX,'DiffMaxChange',DiffMaxChange,'DiffMinChange',DiffMinChange);
            T1map(l,:) = T1; % insert value to T1map
            M0map(l,:) = M0; % insert value to M0map
            normM0map(l,:) = normM0; % insert value to normM0map
            Nmap(l) = N; 
            Errmap(l) = Err;
            MSEmap(l) = MSE;
            PredictedMmap(l,:) = PredictedM;
        end
    end
    
    % Find actual maximum number of components calculated and remove empty
    % components:
    ActualNmax = max(Nmap);
    T1map(:,ActualNmax+1:end)=[];
    M0map(:,ActualNmax+1:end)=[];
    normM0map(:,ActualNmax+1:end)=[];
   
% reshaping T1map, M0map and normM0map to a 4D matrix, and Nmap to a 3d
% matrix:
    T1map = reshape(T1map,[X,Y,Z,ActualNmax]);
    M0map = reshape(M0map,[X,Y,Z,ActualNmax]);
    normM0map = reshape(normM0map,[X,Y,Z,ActualNmax]);
    Nmap = reshape(Nmap,[X,Y,Z]);
    Errmap = reshape(Errmap,[X,Y,Z]);
    MSEmap = reshape(MSEmap,[X,Y,Z]);
    PredictedMmap = reshape(PredictedMmap,[X,Y,Z,IR]);
    IRmask = reshape(mask,[X,Y,Z]);
    
 end
 