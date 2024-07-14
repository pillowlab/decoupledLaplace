function [wmap,mstruct,Hpost] = compMAPwts_bernoulliGLM(xx,yy,varprior,optimopts)
% [wmap,mstruct,Hpost] = compMAPwts_bernoulliGLM(xx,yy,varprior,optimopts)
%
% Compute the MAP weights under Bernoulli GLM given zero-mean Gaussian
% prior with variance 'varprior'.
%
% Inputs
% -------
%       xx [T,d] - design matrix
%       yy [T,1] - binary outputs (0 or 1)
% varprior [1,1] - prior variance
%      optimopts - struct with optimization params (set using optimoptions)
%
% Outputs
% -------
%    wmap [d,1] - MAP estimate of weights
%       mstruct - model structure (with fields for likelihood & prior)
%   Hpost [d,d] - Hessian of negative log posterior


% ======= parse inputs =============

nw = size(xx,2);  % number of weights
w0 = randn(nw,1)*.1; % initial weights

% Set optimization parameters for fminunc
if nargin < 4
    optimopts = optimoptions('fminunc','algorithm','trust-region',...
        'SpecifyObjectiveGradient',true,'HessianFcn','objective',...
        'FunctionTolerance',1e-10,'display','off');
end

% ======= Set up optimization =============


% Make struct with log-likelihood and prior function pointers 
mstruct.neglogli = @neglogli_bernoulliGLM;  % neg-logli function handle
mstruct.logprior = @logprior_stdnormal;         % log-prior function handle
mstruct.liargs = {xx,yy};    % arguments for log-likelihood
mstruct.priargs = {};        % extra arguments for prior (besides theta)

% Set negative log-posterior function handle
lfunc = @(w)(neglogpost_GLM(w,varprior,mstruct));

% ========  Compute MAP estimate of w ===========

wmap = fminunc(lfunc,w0,optimopts);

% ========  Compute Hessian of negative log posterior, if desired ======= 

if nargout > 2

    [~,~,Hpost] = lfunc(wmap);
end
