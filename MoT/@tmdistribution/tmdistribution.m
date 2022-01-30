classdef tmdistribution < classreg.learning.internal.DisallowVectorOps


 properties(GetAccess=public, SetAccess=protected, Hidden =true)
      NDimensions = 0; 
       DistName = 'mixture of t-distributions';
       NComponents = 0;  % number of mixture components
      
       SharedCov = [];
       Iters = [];       % The number of iterations
       RegV = 0;
       CovType = [];
 end
 
 
 properties(GetAccess=public, SetAccess=protected, Dependent=true)

     
   

        DistributionName
         



 end
 
 properties(GetAccess='public', SetAccess='public')      

        mu = [];        
        

        phi = [];

        Sigma = [];       % Covariance
        nu = [];

       
       NumComponents

    

     NumIterations
     
        AIC = [];         % Akaike information criterion

        BIC = [];         % Bayes information criterion


        Converged = [];   % Has the EM converged       
 
       LogLikelihood=[];         % log-likelihood
 

 end
   
   methods
       
       function n = get.DistributionName(this)
           n = this.DistName;
       end
       
       function nc = get.NumComponents(this)
           nc = this.NComponents;
       end

       

       

   end
   
    methods
        function obj = tmdistribution(mu, Sigma, phi,nu)

            if nargin==0
                return;
            end

            if nargin < 2
                error(message('stats:gmdistribution:TooFewInputs'));
            end
            if ~ismatrix(mu) || ~isnumeric(mu)
                error(message('stats:gmdistribution:BadMu'));
            end

            [k,d] = size(mu);
            if ~all(size(Sigma)==size(mu))
                error('mu and Sigma must be of the same length')
            end
            if nargin < 3 || isempty(phi)
                phi = ones(k,1);
            elseif ~isvector(phi) || length(phi) ~= k
                error(message('stats:gmdistribution:MisshapedMuP'));
                     
            elseif any(phi <= 0)
                error(message('stats:gmdistribution:InvalidP'));
                      

            end

            phi = phi/sum(phi);

            [~,sortInd] = sort(mu);
            obj.NDimensions = d;
            obj.NComponents = k;
            obj.phi = phi(sortInd);
            obj.mu = mu(sortInd);
            obj.Sigma = Sigma(sortInd);
            obj.nu = nu(sortInd);
        end % constructor
    end

    methods(Static = true,Hidden)
        obj = fit(X,k,varargin);
    end

    methods(Hidden = true, Static = true)
        function a = empty(varargin)
            throwUndefinedError();
        end
    end
   
end % classdef

 
function throwUndefinedError()
st = dbstack;
name = regexp(st(2).name,'\.','split');
error(message('stats:gmdistribution:UndefinedFunction', name{ 2 }, mfilename));
end

