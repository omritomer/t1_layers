function tmdist = fittmdistnp(x,K,varargin)
% tmdist = fittmdist(x,K,varargin)
% this is a beta version of fittmdist - more detailed documentation to come
% 
rng('default');
rng(1);

% set default algorithm parameters
TolEM = 1e-6; % tolerance for EM algorithm
NuTolX = 1e-6; % x tolerance for optimization of nu parameter
NuTolFun = 1e-6; % function tolerance for optimization of nu parameter
NuDiffMaxChange = 2; % max change allowed for nu in each step
NuDiffMinChange = 0.0001; % min change allowed for nu in each step
NuMin = 2; % min value of nu
NuMax = 40; % max value of nu

% mu_tol = 1;
% Sigma_tol = 10;
% nu_tol = 0.5;
% phi_tol = 0.01;
MaxIter = 500; % maximum number of iterations
Start = 'kmeans'; % default start options
ClusteringStart = 'plus'; % default start points method
fitKernel = false; % whether to apply smoothing to the data by fitting a kernel distribtuion

if rem(length(varargin),2) ~= 0
    error('Incorrect number of input arguments. Name-Value must be in pairs.')
end
for ind=1:2:length(varargin) % process varargin options
    switch(lower(varargin{ind}))
        case 'start'
            Start = varargin{ind+1};
        case 'clusteringstart'
            ClusteringStart = varargin{ind+1};
        case 'maxiter'
            MaxIter = varargin{ind+1};
        case 'tolem'
            TolEM = varargin{ind+1};
        case 'nutolx'
            NuTolX = varargin{ind+1};
        case 'nutolfun'
            NuTolFun = varargin{ind+1};
        case 'nudiffmaxchange'
            NuDiffMaxChange = varargin{ind+1};
        case 'nudiffminchange'
            NuDiffMinChange = varargin{ind+1};
        case 'numin'
            NuMin = varargin{ind+1};
        case 'numax'
            NuMax = varargin{ind+1};
        case 'options'
            options = varargin{ind+1};
        case 'fitkernel'
            fitKernel = varargin{ind+1};
        otherwise
            error('Unrecognized Name-Value pairing')
    end
end
if (exist('options','var')~=1) % If no options struct exists, create it
    options = optimset('TolFun',NuTolFun,'TolX',NuTolX,'DiffMaxChange',NuDiffMaxChange,'DiffMinChange',NuDiffMinChange,'Display', 'off'); % set optimiazation options for nu-estimation
end

if fitKernel % smooth histogram by fitting a kernel distribution to data and then drawing dummy data from it:
    x_old = x;
    pd = fitdist(x,'kernel'); % create kernel distribution
    x = random(pd,[250000 1]); % draw dummy data
end
% Initialize parameters:
N = length(x); % get length of data
mu = zeros(K,1);
Sigma = zeros(K,1);
phi = zeros(K,1);
nu = 15 .* ones(K,1);

if strcmpi(Start,'kmeans') || strcmpi(Start,'k-means')
    if strcmpi(ClusteringStart,'plus')
        [idx,~] = kmeans(x,K,'MaxIter',1000,'Start',ClusteringStart,'Replicates',5); % Use k-means clustering for initialization of clusters
    else
        [idx,~] = kmeans(x,K,'MaxIter',1000,'Start',ClusteringStart,'Replicates',size(ClusteringStart,2)); % Use k-means clustering for initialization of clusters
    end
    % Assign initial parameters value for each k cluster
    for k=1:K
        mu(k) = mean(x(idx==k)); % assign cluster mean (mu)
        Sigma(k) = var(x(idx==k)); % assign cluster variance (Sigma)
        phi(k) = sum(idx==k)./N; % assign mixture weight (phi)
        
        % Estimate degrees-of-freedom parameter (nu)
        sigma = sqrt(Sigma(k));
        M = (x(idx==k)-mu(k))./sigma;
        delta = M.^2; % calculate delta function: delta(x,mu; Sigma) = ((x-mu)^T)*(Sigma^(-1))*(x-mu)
        u = (1 + nu(k)) ./ (delta + nu(k)); % calculate scaling weight u
        fun = @(y)estimateNu(y,u,'init',sum(idx==k)); % create function handle
        nu(k) = lsqnonlin(fun, nu(k),NuMin,40, options); % Solve derivative of the function for degrees-of-freedom equation
        
        % Calculate probability for data to belong to the distribution
        Lk(:,k) = phi(k) .* ...
            ((gamma((nu(k)+1)/2))./(gamma(nu(k)/2)*sqrt(pi*nu(k)*Sigma(k)))) .*...
            ((1 + (1/nu(k)) .* (((x-mu(k)).^2)./Sigma(k))).^(-(nu(k)+1)/2));
        
        %pdf('tLocationScale',x,mu(k),sqrt(Sigma(k)),nu(k));
    end
else
    for k=1:K
        mu(k) = Start.mu(k); % assign cluster mean (mu)
        Sigma(k) = var(x)./(K^2); % assign cluster variance (Sigma)
        phi(k) = 1./K; % assign mixture weight (phi)
        
        % Estimate degrees-of-freedom parameter (nu)
        sigma = sqrt(Sigma(k));
        M = (x-mu(k))./sigma;
        delta = M.^2; % calculate delta function: delta(x,mu; Sigma) = ((x-mu)^T)*(Sigma^(-1))*(x-mu)
        u = (1 + nu(k)) ./ (delta + nu(k)); % calculate scaling weight u
%         fun = @(y)estimateNu(y,u,'init',sum(idx==k)); % create function handle
        nu(k) = 15; %lsqnonlin(fun, nu(k),NuMin,40, options); % Solve derivative of the function for degrees-of-freedom equation
        try
           Sigma(k) = Start.Sigma(k);
           nu(k) = Start.nu(k);
           phi(k) = Start.phi(k);
        end
           
        % Calculate probability for data to belong to the distribution
        Lk(:,k) = phi(k) .* ...
            ((gamma((nu(k)+1)/2))./(gamma(nu(k)/2)*sqrt(pi*nu(k)*Sigma(k)))) .*...
            ((1 + (1/nu(k)) .* (((x-mu(k)).^2)./Sigma(k))).^(-(nu(k)+1)/2));
        
        %pdf('tLocationScale',x,mu(k),sqrt(Sigma(k)),nu(k));
    end
end


LogL = sum(log(sum(Lk,2))); % calculate log-likelihood function


% Begin EM algorithm
for iter=1:MaxIter
    oldLogL = LogL; % log-likelihood of previous step
    
    %% E-step: evaluating the responsibilities using the current parameter values
    tau = Lk ./ repmat(sum(Lk,2),1,K);
    
    %% M-step: re-estimate the parameters using the current responsibilities
    for k = 1:K
        
        phi(k) = sum(tau(:,k))./N; % re-estimate phi
        
        % Calculate scaling weight u
        sigma = sqrt(Sigma(k));
        M = (x-mu(k))./sigma;
        % M is the normalised innovation and M(:,i)'*M(:,i) gives the Mahalanobis
        % distance for each x(:,i).
        delta = M.^2;
        u = (1 + nu(k)) ./ (delta + nu(k));
        
        % Re-estimate mu and Sigma:
        mu(k) = sum(tau(:,k).*u.*x)./sum(tau(:,k).*u); %sum(bsxfun(@times, x, w)) ./ sum(w);
        Sigma(k) = sum(tau(:,k).* u .* ((x - mu(k)).^2))./sum(tau(:,k));
        
        % CM-1 Step
        % ML estimates of mu, S
        
        
        sigma = sqrt(Sigma(k));
        
        % line search is slow so only do it every other iteration
        
        % E step again
        M = (x-mu(k))./sigma;
        % M is the normalised innovation and M(:,i)'*M(:,i) gives the Mahalanobis
        % distance for each x(:,i)
        delta = M.^2;
        u = (1 + nu(k)) ./ (delta + nu(k));
        
        % CM-2 Step
        fun = @(y)estimateNu(y,u,'est',tau(:,k));
        nu(k) = lsqnonlin(fun,nu(k),NuMin,NuMax,options); % calculate nu
        
        % Calculate pdf of data
        Lk(:,k) = phi(k) .* ...
            ((gamma((nu(k)+1)/2))./(gamma(nu(k)/2)*sqrt(pi*nu(k)*Sigma(k)))) .*...
            ((1 + (1/nu(k)) .* (((x-mu(k)).^2)./Sigma(k))).^(-(nu(k)+1)/2));
        % pdf('tLocationScale',x,mu(k),sqrt(Sigma(k)),nu(k));
    end
    LogL = sum(log(sum(Lk,2))); % calculate log-likelihood
    
    LogLdiff = (LogL-oldLogL); % calculate change in log-likelihood function
    flag = LogLdiff >= 0 && LogLdiff < TolEM *abs(LogL); % check if the change is smaller than the tolerance
    %  || (all(abs(mu-mu_stored(:,end))<mu_tol) && ...
    %         all(abs(Sigma-Sigma_stored(:,end))<Sigma_tol) && ...
    %         all(abs(phi-phi_stored(:,end))<phi_tol) && ...
    %         all(abs(nu-nu_stored(:,end))<nu_tol));
    %

    
    
    if flag
        break;
    end
    
end
% end
AIC = (2*(4*K-1)) - 2*LogL;
BIC = (log(N)*(4*K-1)) - 2*LogL;

tmdist = tmdistribution(mu,Sigma,phi,nu);
tmdist.LogLikelihood = LogL;
tmdist.AIC = AIC;
tmdist.BIC = BIC;
tmdist.NumIterations = iter;
tmdist.NumComponents = K;
tmdist.Converged = flag;

end

function f = estimateNu(nu, u,mode,tau)

if strcmpi(mode,'init')
    N = tau;
    f = -psi(nu/2) + log(nu/2) + (sum(log(u)-u)/N) + 1 ...
        + psi((1+nu)/2) - log((1+nu)/2);
elseif strcmpi(mode,'est')
    
    f = -psi(nu/2) + log(nu/2) + (sum(tau .* (log(u)-u))./sum(tau)) + 1 ...
        + psi((1+nu)/2) - log((1+nu)/2);
end

end

