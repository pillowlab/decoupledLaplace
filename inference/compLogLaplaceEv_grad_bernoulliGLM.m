function [dlogEv,dlogEv_terms] = compLogLaplaceEv_grad_bernoulliGLM(theta,xx,yy)
% dlogEv = compLogLaplaceEv_grad_bernoulliGLM(theta,xx,yy)
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
%    theta [1 x 1] - regression weights
%       xx [N x d] - design matrix (each column is a different regressor)
%       yy [N x 1] - vector of binary responses (0 or 1)
%
% Outputs:
% --------
% dlogEv [1 x 1] - derivative of the log-evidence w.r.t. scalar theta
% dlogEv_terms [1 x 3] - derivatives of log-li, log-pri, log post

nw = size(xx,2);  % number of weights

% -----------------------------
% Compute MAP estimate of weights
% -------------------------------
[wmap,~,Hpost,dlogli_dw,dlogpri_dw] = compMAPwts_bernoulliGLM(xx,yy,theta);  % map estimate given theta

% ---------------------------------------------
% Compute gradient of wmap:  H^{-1}  w_map / theta^2
% ---------------------------------------------
dwmap_dthet = (Hpost \ wmap)/theta^2;

% -----------------------------------------------
% 1. Gradient of log-likelihood term log P(Y|x,w) 
% -----------------------------------------------

dlogLi_dtheta = dlogli_dw'*dwmap_dthet; % gradient: (dL/dw) * (dw / dtheta)

% -----------------------------------------------
% 2. Gradient of log-prior term
% -----------------------------------------------

%%% indirect term: dp / dwmap * dwmap / dtheta %%%%
dlogpri_indirect = dlogpri_dw'*dwmap_dthet; 
%%% direct term: dp / dtheta %%%%
dlogpri_direct =  wmap'*wmap/(2*theta^2) - nw/(2*theta); % dp / dtheta term

%%%% sum them up %%%%%
dlogPri_dtheta =  dlogpri_indirect + dlogpri_direct;

% -----------------------------------------------
% 3. Gradient of log-posterior term
% -----------------------------------------------

%%% Direct term %%%%
dlpost_direct = -0.5*trace(inv(Hpost))/theta.^2;

%%% Indirect term %%%%%%%%%%
dHli = compHessDeriv_bernoulliGLM(wmap,xx,dwmap_dthet);
dlpost_indirect = 0.5*sum(sum(inv(Hpost).*dHli));

%%%% sum them up %%%%%
dlogPost_dtheta = dlpost_direct + dlpost_indirect;

% -----------------------------------------------
% Sum up the 3 terms
% -----------------------------------------------
dlogEv = dlogLi_dtheta + dlogPri_dtheta - dlogPost_dtheta;

if nargout > 2
    dlogEv_terms = [dlogLi_dtheta, dlogPri_dtheta, dlogPost_dtheta];
end
