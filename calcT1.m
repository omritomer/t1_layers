function [T1,M0,normM0,residual,SSE,NumOfComponents,PredictedM,MSE] = calcT1(data,TIlist,Nmax,varargin)
%Calculate the T1 components of a voxel
%
%   This function calculates the T1 components of a voxel using nonlinear
%   least-squares optimization with a trust-region-reflective algorithm of
%   a voxel containing multiple inversion recovery data. The function uses
%   a chi-squared difference test (Schermelleh-Engel, Moosbrugger & M?ller,
%   2003) to determine the number of T1 components to fit to the data. The
%   data is fitted to the inversion recovery (IR) equation:
%           M(i) = {sum_from_N1_to_Nmax}{M0(N)*(1-2^(-TI(i)/T1(N)))}
%   where M(i) is the i-th element of data and TI(i) is the i-th element of
%   TIlist.
%
%
% Syntax:
%   [T1,M0,normM0] = calcT1(data,TIlist,Nmax)
%   [T1,M0,normM0] = calcT1(data,TIlist,Nmax,Name,Value)
%   [T1,M0,normM0,residual,SS,NumOfComponents] = calcT1(_______)
%
%
% Description:
%   [T1,M0,normM0] = calcT1(data,TIlist,Nmax)of  returns a vector of up to
%   Nmax T1-components of the voxel (T1) fitted to the inversion recovery
%   equation, with data and TIlist being the supplied parameters. The
%   function also outputs vector of the matching initial magnetization of
%   each value in T1 (M0), and a vector of the normalized values of M0
%   (normM0), so that sum(normM0) == 1.
%
%   [T1,M0,normM0] = calcT1(data,TIlist,Nmax,Name,Value) also supplies
%   different Name-Value arguments related to the optimization (for more
%   see below).
%
%   [T1,M0,normM0,residual,SS,NumOfComponents] = calcT1(_______) also
%   returns the residual, the sum of squares (SS) and the actual number of
%   componenets fitted to the data (NumOfComponents).
%
%
% Input arguments:
%
%   data - a vector with K elements corresponding to the signal in K
%   inversion times.
%   TIlist - a vector with K elements corresponding to the inversion times.
%   Nmax - a scalar representing the maximum number of components to fit to
%   the data.
%   Name-Value pairings:
%          pVal - threshold p-value for model comparison between simpler
%          model and more complicated model. If the chi-squared difference
%          test is significant, the more complicated model is chosen.
%          Otherwise, the simpler model is chosen.
%          Display - level of display (default: 'off'). For more see
%          lsqnonlin.
%          Diagnostics - level of display (default: 'off'). For more see
%          lsqnonlin.
%          TolFun - termination tolerance on the function value, a positive
%          scalar. For more see option *FunctionTolerance* in lsqnonlin
%          (default: 0.1).
%          TolX - termination tolerance on x, a positive scalar. For more
%          see option *StepTolerance* in lsqnonlin (default: 0.1).
%          DiffMaxChange - maximum change in variables for finite
%          difference gradients (a positive scalar). For more see lsqnonlin
%          (default: 0.1).
%          DiffMinChange - minimum change in variables for finite
%          difference gradients (a positive scalar). For more see lsqnonlin
%          (default: 0.0001).
%
% Output arguments:
%
%   T1 - a 1-by-Nmax vector of the T1 components fitted to the voxel. If
%   the actual number of components fitted is less than Nmax, the vector is
%   zero-padded.
%   M0 - a 1-by-Nmax vector of the initial magnetization values matching to
%   the T1 values in T1. If the actual number of components fitted is less
%   than Nmax, the vector is zero-padded.
%   normM0 - a 1-by-Nmax vector of the normalized values of M0, so that
%   sum(normM0) == 1. Calculated as normM0 = M0 ./ sum(M0).
%   residual - value of objective function at solution, returned as a
%   vector. For more see: lsqnonlin.
%   SS - Squared norm of the residual, returned as a nonnegative real. For
%   more see argument resnorm in lsqnonlin.
%   NumOfComponents - the number of components that was fitted to the data.

% Set default optimization parameters for lsqnonlin
Display = 'off';
Diagnostics = 'off';
TolFun = 0.0001;
TolX = 0.0001;
DiffMaxChange = 10;
DiffMinChange = 0.0001;
pVal = 0.01; % threshold p-value for model comparison (chi-squared difference test)

% Get Name-Value pairs of optimization parameters.
for ind=1:2:length(varargin);
    switch(lower(varargin{ind}))
        case 'pval'
            pVal =  varargin{ind+1};
        case 'display'
            Display = varargin{ind+1};
        case 'diagnostics'
            Diagnostics = varargin{ind+1};
        case 'tolfun'
            TolFun = varargin{ind+1};
        case 'tolx'
            TolX = varargin{ind+1};
        case 'diffmaxchange'
            DiffMaxChange = varargin{ind+1};
        case 'diffminchange'
            DiffMinChange = varargin{ind+1};
        otherwise
            error(['The parameter ' varargin{ind+1} ' is not recognized by makePmap.']);
            
    end
end

% Set optimization options struct:
 h = optimset('Display',Display,'Diagnostics',Diagnostics,...
     'TolFun',TolFun,'TolX',TolX,...
     'DiffMaxChange',DiffMaxChange,'DiffMinChange',DiffMinChange);
% h = optimset('Display',Display,'Diagnostics',Diagnostics);

data = double(reshape(data,1,[])); % reshape data to row vector and convert to double
TIlist = double(reshape(TIlist,1,[])); % reshape TIlist to row vector and convert to double
K = length(data); % get number of data points
mIR = max(data); % get maximum data points

% Initialize matrices of the calculated components. Each row is a different
% iteration:
T1_mat = zeros(Nmax,Nmax); % matrix of T1 values
M0_mat = zeros(Nmax,Nmax); % matrix of M0 values

% Initialize other variables. Dimensions set to Nmax is for different
% iterations:
PredictedM = zeros(Nmax,K); % Matrix of calculated data values post-optimization
residual_matrix = zeros(Nmax,K); % Residual of optimization
SSE = zeros(1,Nmax); % Sum of squares of optimization
chi_sq = zeros(1,Nmax); % chi square of optimization [sum((M-data).^2/M)]
df = zeros(1,Nmax); % degree of freedom of optimization [K-N]
chi_diff = zeros(1,Nmax-1); % difference between chi-squares of the simpler model and the chi_diff = chi_sq(N-1) - chi_sq(N) 
df_diff = zeros(1,Nmax-1); % df_diff = df(N-1) - df(N) 
p = zeros(1,Nmax-1); % p-value of chi_diff.  p = 1-cdf('Chisquare',chi_diff,df_diff);

N = 2; % Initial number of components
while N<=Nmax
    % The number of components is optimized by using a chi-squared
    % difference test to determine whether adding more model parmaters
    % (i.e. components) adds information that enables a better fit. for
    % more see:
    % Schermelleh-Engel, K., Moosbrugger, H., & M?ller, H. (2003).
    % Evaluating the fit of structural equation models: Tests of
    % significance and descriptive goodness-of-fit measures. Methods of
    % psychological research online, 8(2), 23-74.
    
    x0 = [1200 linspace(600,2500,N-1) (mIR/N)*ones(1,N)]; % inital conditions. x0(1:N) is T1 values, x0(N+1:end) is M0 values 
    lb = zeros(1,2*N); % lower bounds
    ub = 4000*ones(1,2*N); % upper bounds
    % lsqnonlin returns vector x that minimizes the function optT1 between
    % the lower and upper bounds (lb and ub). Also returned are SS_vec 
    % (resnorm) and residual_matrix (residual). For more see lsqnonlin.
    [x,SSE(N),residual_matrix(N,:),exitflag,output,L,J] = lsqnonlin(@optT1,x0,lb,ub,h,data,TIlist,N);
    
    T1_mat(N,1:N) = x(1:N); % get T1 values from x
    M0_mat(N,1:N) = x(N+1:end); % get M0 values from x
    % calculate expected signal for each component:
    M_c = (repmat(M0_mat(N,:)',1,K).*(1-2*exp(-repmat(TIlist,Nmax,1)./repmat(T1_mat(N,:)',1,K))));
    % sum signal across all components:
    PredictedM(N,:) = abs(sum(M_c,1));
    
    % Calculate chi-square statistic of optimization and the degrees of
    % freedom:
    %chi_sq(N) = sum(((M(N,:)-data).^2)./M(N,:));
    df(N) = K - 2*N;
    MSE(N) = SSE(N)./df(N);
    LogL(N) = K.*log(1./sqrt(2.*pi.*MSE(N)))-SSE(N)./(2*MSE(N)^2);
    BIC(N) = log(K)*N-2*LogL(N);
    
    % When there is more than one component, test whether the more
    % complicated model is a better fit with a chi-square difference test.
    % The chi-square statistic is:
    % chi_diff = chi_sq(simpler) - chi_sq(complicated)
    % The degrees of freedom are:
    % df_diff = df_sq(simpler) - df_sq(complicated)
    % The p-value is drawn from the Chi-square distribution
%     if N>1
%         chi_diff(N-1) = -diff(chi_sq(N-1:N)); % calculate chi-square statistic
%         df_diff(N-1) = -diff(df(N-1:N)); % calculate degrees-of-freedom
%         p(N-1) = 1-cdf('Chisquare',chi_diff(N-1),df_diff(N-1)); % calculate p-value
%         if p(N-1)>=pVal % if p-value is not significant, prefer the simpler model
%             % Create output parameters and break loop:
%             T1 = T1_mat(N-1,:);
%             M0 = M0_mat(N-1,:);
%             NumOfComponents = N-1;
%             residual = residual_matrix(N-1,:);
%             SS = SS_vec(N-1);
%             break
%         end
%     end
    N = N+1; % increase number of components
end
% if N > Nmax % if Nmax is reached, use Nmax components:
%     T1 = T1_mat(Nmax,:);
%     M0 = M0_mat(Nmax,:);
%     NumOfComponents = Nmax;
%     residual = residual_matrix(Nmax,:);
%     SS = SS_vec(Nmax);
%     M = M(Nmax,:);
% end
% [~,maxLogL] = max(LogL); 
% [~,minBIC] = min(BIC);
SSE(1) = SSE(2)+1;
[SSE,minSSE] = min(SSE);
T1 = T1_mat(minSSE,:);
M0 = M0_mat(minSSE,:);
NumOfComponents = minSSE;
residual = residual_matrix(minSSE,:);
MSE = MSE(minSSE);
PredictedM = PredictedM(minSSE,:);
normM0 = M0 ./ sum(M0); % normalized values of M0
if any(isnan(normM0))
    T1 = zeros(size(T1));
    M0 = zeros(size(M0));
    normM0 = zeros(size(normM0));
    PredictedM = zeros(size(PredictedM));
    residual = zeros(size(residual));
    MSE = zeros(size(MSE));
    SSE = zeros(size(SSE));
    NumOfComponents = zeros(size(NumOfComponents));
end

end

function [residuals,J] = optT1(x,data,TIlist,N)
%Optimization function of calcT1
%
%   This is the optimization function called by calcT1 for lsqnonlin. For
%   more see calcT1
%
%
% Syntax:
%   residuals = optT1(x,data,TIlist,N)
%
%
% Description:
%   residuals = optT1(x,data,TIlist,N) calculates the residual between the
%   data and theM calculated from x and TIlist.
%
%
% Input arguments:
%
%   x - a vector with 2*N elements. Element 1:N are T1 values, elements
%   N+1:2*N are M0 values
%   data - a vector with K elements corresponding to the signal in K
%   inversion times.
%   TIlist - a vector with K elements corresponding to the inversion times.
%   N - The number of components calculated
%
% Output arguments:
%
%   residuals - the residuals calculated as M-data.

T1 = x(1:N); % get T1 values from x
M0 = x(N+1:end); % get T1 values from x

% calculate expected signal for each component:
M_c = (repmat(M0',1,length(TIlist)).*(1-2*exp(-repmat(TIlist,length(T1),1)./repmat(T1',1,length(TIlist)))));
% sum signal across all components:
M = abs(sum(M_c,1));

% calculate residuals:
residuals = M-data;


end
