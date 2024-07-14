% test_deriveGrad_LogPost.m
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

%% 2. Compute derivative of log laplace evidence estimate of weights given true hyperparams

theta0 = 2;  % prior variance at which to evaluate gradient

% compute gradient using analytic formulas
dlogEv = compLogLaplaceEv_grad_bernoulliGLM(theta0,xx,yy);

%% 3. Compare to finite differencing version

% compute map estimate at theta0
[wmap0,mstruct] = compMAPwts_bernoulliGLM(xx,yy,theta0);  % map estimate given theta0
ev0 = compLogLaplaceEv(theta0,mstruct);

% compute MAP estimate at theta0 + dtheta
dtheta = .01; % change in theta 
theta1 = theta0+dtheta;  % new theta
ev1 = compLogLaplaceEv(theta1,mstruct);

% Compute finite difference
dlogEv_empir = (ev1-ev0)/dtheta;


%% 4. print comparison

fprintf('\nDeriv of log-posterior\n');
fprintf('-----------------------------------------\n');
fprintf(' analytical: %.5f\n', dlogEv);
fprintf('finite-diff: %.5f\n', dlogEv_empir);


