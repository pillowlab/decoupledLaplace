% test_deriveGrad_LogLi_and_Prior.m
%
% Test out computation of gradient of the log-likelihood and log-prior terms in the log-evidence

clear; clf; clc;
addpath utils;
addpath inference;

% 1.  Set up simulated example

% set true weights
nw = 5;         % number of weights
varpriortrue = 1;  % true prior variance of weights
wts = randn(nw,1)*sqrt(varpriortrue); % Sample weights from prior
Iw = eye(nw);

% Make stimuli & simulate Bernoulli GLM response
nstim = 25;           % number of stimuli
xx = randn(nstim,nw); % inputs
xproj = xx*wts;       % projection of stimulus onto weights
pp = logistic(xproj);  % probability of 1
yy = rand(nstim,1)<pp; % Bernoulli outputs

%% 2. Compute MAP estimate of weights given true hyperparams

theta0 = 3;  % prior variance at which to compute initial MAP estimate & Laplace evidence
[wmap0,mstruct] = compMAPwts_bernoulliGLM(xx,yy,theta0);  % map estimate given theta0

%% 3. Compute gradient d wmap / dtheta

% Compute grad and Hessian of log-li
[negL0,dnegL0,Hli0] = neglogli_bernoulliGLM(wmap0,xx,yy);

% Hessian of negative log-prior
Hpri = (1/theta0)*eye(nw);

% Hessian of posterior
Hpost = (Hli0+Hpri);

% Compute gradient of wmap:  H^{-1}  w_map / theta^2
dwmap_dtheta = (Hpost\wmap0)/theta0^2;

%% 4.  Compute gradient of log-likelihood w.r.t. theta

% Analytic formula (simply dL/dw times dw / dtheta):
dL_dtheta = -dnegL0'*dwmap_dtheta;


% Finite differencing
% -------------------
dtheta = .01; % change in theta 
theta1 = theta0+dtheta;  % new theta
wmap1 = compMAPwts_bernoulliGLM(xx,yy,theta1);  % Compute new MAP estimate

% compute value of neg log-likelihood at new wmap
negL1 = neglogli_bernoulliGLM(wmap1,xx,yy);

% compute finite-differencing gradient
dL_dtheta_empir = (-negL1 + negL0)/dtheta;

% Print comparison
% -------------------
fprintf('Deriv of log-likelihood (dL / dtheta)\n');
fprintf('-------------------------------------\n')
fprintf(' analytical: %.5f\n', dL_dtheta);
fprintf('finite-diff: %.5f\n', dL_dtheta_empir);


%% 5. Compute gradient of log-prior w.r.t. theta

% Analytic formula (simply dP/dw times dw / dtheta):
% --------------------------------------------------

% gradient of log-prior w.r.t. w at theta0
[logpri0,dlogpri_dw0] = logprior_stdnormal(wmap0,theta0);

% gradient of log-prior w.r.t. theta
% ------------------------------------
% indirect term: dp / dwmap * dwmap / dtheta
trm1 = dlogpri_dw0'*dwmap_dtheta; 
% direct term: dp / dtheta
trm2 =  wmap0'*wmap0/(2*theta0^2) - nw/(2*theta0); % dp / dtheta term
dlogpri_dtheta =  trm1 + trm2;

% Finite differencing
% -------------------
dtheta = .01; % change in theta 
theta1 = theta0+dtheta;  % new theta
wmap1 = compMAPwts_bernoulliGLM(xx,yy,theta1);  % Compute new MAP estimate

% Compute value of log-prior at new wmap
logpri1 = logprior_stdnormal(wmap1,theta1);

% compute finite-differencing gradient
dlogpri_dtheta_empir = (logpri1-logpri0)/dtheta;

% Print comparison
% -------------------
fprintf('\nDeriv of log-prior (d log prior / dtheta)\n');
fprintf('-----------------------------------------\n');
fprintf(' analytical: %.5f\n', dlogpri_dtheta);
fprintf('finite-diff: %.5f\n', dlogpri_dtheta_empir);

