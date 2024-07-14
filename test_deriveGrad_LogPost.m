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

%% 2. Compute MAP estimate of weights given true hyperparams

theta0 = 2;  % prior variance at which to compute initial MAP estimate & Laplace evidence
[wmap0,~,Hpost0] = compMAPwts_bernoulliGLM(xx,yy,theta0);  % map estimate given theta0

%% 3. Compute gradient d wmap / dtheta

% Compute gradient of wmap:  H^{-1}  w_map / theta^2
dwmap_dtheta = (Hpost0\wmap0)/theta0^2;

%% 4.  Compute gradient of log-posterior w.r.t. theta

% Finite differencing
% ===================

% Evaluate log-post at theta0
% -------------------------------
logpost0 = .5*logdet(Hpost0)-(nw/2)*log(2*pi); % log-posterior at wmap

% Update theta & wmap
% -------------------
dtheta = .01; % change in theta 
theta1 = theta0+dtheta;  % new theta
[wmap1,~,Hpost1] = compMAPwts_bernoulliGLM(xx,yy,theta1);  % Compute new MAP estimate

% Evaluate log-post at theta1
% -------------------------------
logpost1 = .5*logdet(Hpost1)-(nw/2)*log(2*pi); % log-posterior at wmap

% Compute finite diff
% -------------------------------
dlogpost_empir = (logpost1-logpost0)/dtheta;

% Analytic gradient
% =================

%%% Direct term %%%%
dlogpost_direct = -0.5*trace(inv(Hpost0))/theta0.^2;

%%% Indirect term %%%

dH = compHessDeriv_bernoulliGLM(wmap0,xx,dwmap_dtheta);
dlogpost_indirect = 0.5*sum(sum(inv(Hpost0).*dH));

% Sum direct and indirect terms
dlogpost_dtheta = dlogpost_direct + dlogpost_indirect;

% Print comparison
% -------------------
fprintf('\nDeriv of log-posterior\n');
fprintf('-----------------------------------------\n');
fprintf(' analytical: %.5f\n', dlogpost_dtheta);
fprintf('finite-diff: %.5f\n', dlogpost_empir);


% % check numerically that we didn't screw up in the function
% [dlogEv,dlogLi,dlogPri,dlogPost,dwmap_dthet] = compLogLaplaceEv_grad_bernoulliGLM(theta0,xx,yy);
% 
% [dlogpost_analytical,dlogPost]
