function [neglogEv,grad] = neglogEv_NewtonLaplace(logtheta,mstruct,wmap0,ddnegL0,Lmu0) 
% [neglogEv,grad] = neglogEv_NewtonLaplace(logtheta,mstruct,wmap0,ddnegL0,Lmu0) 
%
% Compute Newton-Laplace Approximate Evidence

theta = exp(logtheta);
nw = length(wmap0);

% Compute inverse prior covariance
[~,~,negCinv] = mstruct.logprior(wmap0,theta,mstruct.priargs{:}); 
Cinv_giventheta = -negCinv;

% Compute posterior Hessian using original log-likelihood Hessian
Hess_giventheta = (ddnegL0+Cinv_giventheta);

% Compute updated w_MAP
wmap_giventheta = Hess_giventheta\Lmu0;

% --------------------------------------
% Compute prior terms
logp_ale = mstruct.logprior(wmap_giventheta, theta,mstruct.priargs{:});

% --------------------------------------
% Compute updated log-likelihood and Hessian 
[negL,~,ddnegL1] = mstruct.neglogli(wmap_giventheta,mstruct.liargs{:});
Hess_updated = (ddnegL1+Cinv_giventheta);

% --------------------------------------
% Compute posterior term
logpost_aleNewton = + .5*logdet(Hess_updated) - nw/2*log(2*pi);

% --------------------------------------
% Sum up terms to get ALE
neglogEv = negL - logp_ale + logpost_aleNewton;
    
grad = 0;
