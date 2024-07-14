function dHdtheta = compHessDeriv_bernoulliGLM(wts,X,dwdtheta)
% dHdtheta = compHessDeriv_bernoulliGLM(wts,X,Y,dwdtheta)
%
% Compute derivative of Hessian with respect to a scalar parameter theta
% given the vector dw / dtheta
%
% This derivative can be expressed in two equivalent ways:
% 1. dH / dtheta = dH/dw * dw/dv 
% 2. dH / dtheta = d/dr H(w + r*dwdv) evaluated at r=0.
%
% This function uses the 2nd of these formulas by explicitly evaluating 
%  d/dr H(w + r dwdv).
% 
% Inputs:
% -------
%      wts [d x 1] - regression weights
%        X [N x d] - design matrix (each column is a different regressor)
% dwdtheta [d x 1] - vector dw / dtheta
%
% Outputs:
% --------
% dHdtheta [1 x 1] - derivative of Hessian matrix X w.r.t. scalar theta

xproj = X*wts;       % project inputs onto GLM weights
pp = 1./(1+exp(-xproj)); % evaluate probabilities of class assignments

dxdw = X*dwdtheta;  % project inputs onto vector dwdtheta
alphas = pp.*(1-2*pp).*(1-pp).*dxdw; % weights for each x_i x_i^T 

% Compute weighted outer products of stimuli
dHdtheta = X'*(X.*alphas); 


