function [logp,dlogp,ddlogp] = logprior_stdnormal(ww,vr,iirdge,vr0)
% [logp,dlogp,ddlogp] = logprior_stdnormal(xx,vr,iirdge,vr0)
%
% Evaluate log-pdf of zero-mean Gaussian at parameter vector ww:
%
%  w ~ N(0,vr)
%
%  log P(w) = - 0.5 x^T Cinv x - 0.5 logdet(2*pi*C)
%  
%  where C = vr*I
%
% Inputs:
%      ww [n x 1] - parameter vector (last element can be DC)
%      vr [1 x 1] - prior ariance 
%  iirdge [v x 1] - indices to apply ridge prior to (optional)
%  vrNull [1 x 1] - prior variance for other elements
%
% Outputs:
%       logp [1 x 1] - log-prior: - 0.5 x^T Cinv x - 0.5 logdet(2*pi*C)
%      dlogp [n x 1] - gradient:  - Cinv x
%     ddlogp [n x n] - Hessian:   - Cinv  

% Get size of ww
[nw,nvecs] = size(ww);

% Determine if only some
if (nargin < 3) || isempty(iirdge)
    iirdge = (1:nw)';
    vr0 = 1;
elseif (nargin < 4) || isempty(vr0)
    vr0 = 1;
end

% Build diagonal inverse cov
Cinvdiag = (1/vr0)*ones(nw,1);
Cinvdiag(iirdge) = 1/vr;

logdetterm = 0.5*sum(log(Cinvdiag/(2*pi)));

% Compute log-prior and gradient
if (nvecs == 1)
    dlogp = -ww.*Cinvdiag; % grad
    logp = .5*dlogp'*ww + logdetterm; % logli
else
    % If multiple 'prvec' vectors passed in
    dlogp = -bsxfun(@times,ww,Cinvdiag); % grad vectors
    logp = .5*sum(bsxfun(@times,dlogp,ww),1)+logdetterm; % logli values
end

% Compute Hessian, if desired
if nargout > 2
    ddlogp = spdiags(-Cinvdiag,0,nw,nw);
end
