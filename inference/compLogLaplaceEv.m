function [logEv,wmap,postHess] = compLogLaplaceEv(theta,mstruct,w0,optimopts)
% [logEv,wmap,postHess] = compLogLaplaceEv(theta,mstruct,w0,optimopts)


% ========  Parse inputs ===========

% intial guess for weights (random)
if nargin < 3 
    nw = size(mstruct.liargs{1},2);  % get number of weights
    w0 = randn(nw,1)*.1;
elseif isempty(w0)
    nw = size(mstruct.liargs{1},2);  % get number of weights
    w0 = randn(nw,1)*.1;
end

if nargin < 4
    % Set optimization parameters for fminunc
    optimopts = optimoptions('fminunc','algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective');
end

% ========  Compute MAP estimate of w ===========

% Set negative log-posterior function handle
lfunc = @(w)(neglogpost_GLM(w,theta,mstruct));

% Compute MAP estimate
wmap = fminunc(lfunc,w0,optimopts);


% ========  Evaluate log-evidence using Laplace approx ===========

% evaluate log-likelihood and its Hessian
[negL,~,ddnegL] = mstruct.neglogli(wmap,mstruct.liargs{:}); 

% evaluate log-prior and its Hessian
[logpri,~,negCinv] = mstruct.logprior(wmap,theta,mstruct.priargs{:});

% Hessian of posterior
postHess = ddnegL - negCinv;

% log-posterior
logpost = .5*logdet(postHess) - length(wmap)/2*log(2*pi);

% Compute log-evidence using Laplace approximation
logEv = (-negL) + logpri - logpost;