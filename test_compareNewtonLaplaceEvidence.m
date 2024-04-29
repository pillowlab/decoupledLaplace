% test_compareNewtonLaplaceEvidence.m
%
% Tests out an alternative to decoupled Laplace called (for now)
% Newton-Laplace.  In this case, we update the MAP estimate "wmap" just as we
% do in decoupled Laplace.  Then we evaluate the log-likelihood Hessian at
% this new value of wmap. 
%
% Two discoveries:
% 1. Our closed-form update to wmap is equivalent to a single newton step.
% 2. The new method seems to fix gradient issues with decoupled Laplace.

% 1.  Set up simulated example
addpath utils;

% Set dimensions and hyperparameter
varprior = 2.5;  % prior variance of weights
nw = 50;         % number of weights
nstim = 250;     % number of stimuli
vlims = log10([.1, 4]); % limits of grid over sig^2 to consider
theta0 = 1; % prior variance for DLA

% Sample weights from prior
wts = randn(nw,1)*sqrt(varprior);

% Make stimuli & simulate Bernoulli GLM response
xx = randn(nstim,nw); % inputs
xproj = xx*wts;       % projection of stimulus onto weights
pp = logistic(xproj);  % probability of 1
yy = rand(nstim,1)<pp; % Bernoulli outputs

%% 2. Compute MAP estimate of weights given true hyperparams

% Make struct with log-likelihood and prior function pointers 
mstruct.neglogli = @neglogli_bernoulliGLM;  % neg-logli function handle
mstruct.logprior = @logprior_stdnormal;         % log-prior function handle
mstruct.liargs = {xx,yy};    % arguments for log-likelihood
mstruct.priargs = {};        % extra arguments for prior (besides theta)

% Set optimization parameters for fminunc
opts = optimoptions('fminunc','algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective','display','off');

% intial guess for weights (random)
w0 = randn(nw,1)*.1;

% make function handle
lfunc = @(w)(neglogpost_GLM(w,varprior,mstruct));
% % Optional: Check that analytic gradient and Hessian are correct
% HessCheck(lfunc, w0);

% Compute MAP estimate
[wmap,neglogpost] = fminunc(lfunc,zeros(nw,1),opts);

% Make Plot 
subplot(211); 
tt = 1:nw; % grid of coefficient indices
plot(tt,wts,tt,wmap);
title('true weights and MAP estimate'); box off;
xlabel('coefficient #'); ylabel('weight');
legend('true weights', 'MAP estim'); 


%% 3. Evaluate Laplace Evidence on a grid

% set of grid values to consider
ngrid = 25; % number of grid points
vargrid = logspace(vlims(1),vlims(2),ngrid);

% allocate storage 
logLaplaceEv = zeros(ngrid,1); 

for jj = 1:ngrid
    logLaplaceEv(jj) = compLogLaplaceEv(vargrid(jj),mstruct,wmap,opts);
end

% Find maximum (from grid values);
[logLaplEvMax,ivarHat]=max(logLaplaceEv);
varHat = vargrid(ivarHat);

subplot(212);
plot(vargrid,logLaplaceEv,varHat,logLaplEvMax,'*');
xlabel('sig^2'); ylabel('log-evidence');
title('log-evidence vs. theta'); box off;

%theta0=varHat

%% 4. Now variance approximate evidence functions on the same grid

% First, compute MAP estimate given this value of theta
lfunc = @(w)(neglogpost_GLM(w,theta0,mstruct));
wmap0 = fminunc(lfunc,zeros(nw,1),opts);  % get MAP estimate

% Compute gradient and Hessian of negative log-likelihood 
[negL0,dnegL0,ddnL0] = mstruct.neglogli(wmap0,mstruct.liargs{:}); 

% Compute Hessian of log-prior 
[logp,~,negCinv] = mstruct.logprior(wmap0,theta0,mstruct.priargs{:});

% Compute log-evidence using Laplace approximation
postHess0 = ddnL0-negCinv; % posterior Hessian
logpost = .5*logdet(postHess0)-(nw/2)*log(2*pi); % log-posterior at wmap
logEv0 = (-negL0) + logp - logpost;  % log-evidence

% Compute Hessian of negative log-likelihood times log-likelihood mean
ddnLmu0 = postHess0*wmap0;

% Compute some constants
logpriconst = - nw/2*log(2*pi);   % normalizing constant for log prior 
norm2wmap0 = sum(wmap0.^2); % squared L2 norm of wmap0 

% allocate storage for approximate Laplace Evidence (ALE)
logALE_moving = zeros(ngrid,1); 
logALE_fixed = zeros(ngrid,1); 
logALE_movingNewton = zeros(ngrid,1); 
logALE_fixedNewton = zeros(ngrid,1); 

for jj = 1:ngrid

    % make inverse prior covariance
    Cinv_giventheta = (1/vargrid(jj))*eye(nw);  % inverse prior covariance
    logdetCinv = -nw*log(vargrid(jj)); % log-determinant of inv prior cov
    
    % Compute updated posterior Hessian
    Hess_giventheta = (ddnL0+Cinv_giventheta);
    
    % Compute updated w_MAP
    wmap_giventheta = Hess_giventheta\ddnLmu0;

    % =======================
    % moving ALE
    % =======================
    
    % Compute log prior 
    logp_moving = -.5*sum(wmap_giventheta.^2)/vargrid(jj)+ .5*logdetCinv + logpriconst;
    % Compute negative log-likelihood 
    negL_moving = mstruct.neglogli(wmap_giventheta,mstruct.liargs{:});  
     % Compute log posterior 
    logpost_moving = .5*logdet(Hess_giventheta) + logpriconst; % (note quadratic term is 0)

    % Compute ALE (moving)
    logALE_moving(jj) = -negL_moving + logp_moving - logpost_moving;
    
    % =======================
    % fixed ALE
    % =======================

    % Compute prior term
    logp_fixed = -.5*norm2wmap0/vargrid(jj)+ .5*logdetCinv + logpriconst;
    % Compute posterior term
    logpost_fixed = -0.5*sum((wmap0-wmap_giventheta).^2) ...
        + .5*logdet(Hess_giventheta)+logpriconst;
    
    % Compute ALE (fixed)
    logALE_fixed(jj) = -negL0 + logp_fixed - logpost_fixed;
    
    % ===========================================================
    % Compute Newton-Laplace ALE
    % ===========================================================

    % Compute updated Hessian for log-likelihood using new wmap
    [~,~,ddnL1] = mstruct.neglogli(wmap_giventheta,mstruct.liargs{:});
    Hess_updated = (ddnL1+Cinv_giventheta);

    % ================
    % moving-Newton ALE
    % ================
    
     % Compute log posterior (moving)
    logpost_movingNewton = .5*logdet(Hess_updated) + logpriconst; % (note quadratic term is 0)

    % Compute ALE (Newton-moving)
    logALE_movingNewton(jj) = -negL_moving + logp_moving - logpost_movingNewton;
    % Compute ALE (Newton-moving)
    
    % ================
    % fixed-Newton ALE
    % ================

    % Compute posterior term
    logpost_fixedNewton = -0.5*sum((wmap0-wmap_giventheta).^2) ...
        + .5*logdet(Hess_updated)+logpriconst;
    
    % Compute ALE (Newton-fixed)
    logALE_fixedNewton(jj) = -negL0 + logp_fixed - logpost_fixedNewton;
    
end

% Make plot of LE and ALE
subplot(212);
plot(vargrid,logLaplaceEv,vargrid,logALE_moving,vargrid,logALE_fixed, ...
    vargrid,logALE_fixedNewton,'--',vargrid,logALE_movingNewton,'--',...    
    theta0,logEv0,'ko',varHat,logLaplEvMax,'*');
xlabel('variance (sig^2)'); ylabel('log-evidence');
title('log-evidence vs. precision'); box off;
legend('Laplace Evidence', 'ALE (moving)', 'ALE (fixed)', 'newALE (moving)', ...
    'newALE (fixed)', 'theta_0','theta max'); 
set(gca,'ylim',[min(logLaplaceEv)-1,max([logALE_moving;logLaplaceEv])+1]);
