% test_deriveGradLogEvidence.m
%
% Tests out explicit derivation of the gradient of the Laplace
% approximation based log-evidence. 
% this new value of wmap. 

clear; clf;
addpath utils;
addpath inference;

% 1.  Set up simulated example

% set true weights
nw = 10;         % number of weights
varpriortrue = 3;  % true prior variance of weights
wts = randn(nw,1)*sqrt(varpriortrue); % Sample weights from prior
Iw = eye(nw);

% Make stimuli & simulate Bernoulli GLM response
nstim = 20;           % number of stimuli
xx = randn(nstim,nw); % inputs
xproj = xx*wts;       % projection of stimulus onto weights
pp = logistic(xproj);  % probability of 1
yy = rand(nstim,1)<pp; % Bernoulli outputs

%% 2. Compute MAP estimate of weights given true hyperparams

theta0 = 5;  % prior variance at which to compute initial MAP estimate & Laplace evidence
wmap0 = compMAPwts(xx,yy,theta0);  % map estimate given theta0

%% 3. Compute gradient d wmap / dtheta using analytic formula

% Compute grad and Hessian of log-li
[~,dnegL,Hli] = neglogli_bernoulliGLM(wmap0,xx,yy);

% Hessian of negative log-prior
Hpri = (1/theta0)*eye(nw);

% Hessian of posterior
Hpost = (Hli+Hpri);

% Compute gradient of wmap:  H^{-1}  w_map / theta^2
dwmap_dtheta = (Hpost\wmap0)/theta0^2;


%% 4. Compute gradient d wmap / dtheta using finite differencing

dtheta = .01; % change in theta 
theta1 = theta0+dtheta;
wmap1 = compMAPwts(xx,yy,theta1); % Compute new MAP estimate

% finite differencing formula for gradient
dwmap_dtheta_empir = (wmap1-wmap0)/dtheta;


%% 4. compare the analytic and finite differencing gradient

plot(1:nw,dwmap_dtheta,'-o',1:nw,dwmap_dtheta_empir,'--o');
hold on;, plot([1 nw], [0 0], 'k', 'LineWidth',1); hold off;
xlabel('weight #');
ylabel('dw/dtheta')
title('analytic and empirical gradient dwmap / dtheta'); 
box off;
legend('analytic','finite diff');

