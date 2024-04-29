% test_compareDLAandNewtonWupdate.m
%
% Verify numerically that closed-form update to wmap under DLA is
% equivalent to a single Newton step 

% Set up simulated example
addpath utils;

% Set dimensions and hyperparameter
varprior = 2.5;      % prior variance of weights
nw = 10;            % number of weights
nstim = 250;       % number of stimuli

theta0 = 1; % prior variance for initial Laplace approx
theta1 = 2; % prior variance for next point in hyperparameter space

% Sample weights from prior
wts = randn(nw,1)*sqrt(varprior);

% Make stimuli & simulate Bernoulli GLM response
xx = randn(nstim,nw); % inputs
xproj = xx*wts;       % projection of stimulus onto weights
pp = logistic(xproj);  % probability of 1
yy = rand(nstim,1)<pp; % Bernoulli outputs

%% 2. Compute MAP estimate of weights at sig^2 = theta0

% Make struct with log-likelihood and prior function pointers 
mstruct.neglogli = @neglogli_bernoulliGLM;  % neg-logli function handle
mstruct.logprior = @logprior_stdnormal;         % log-prior function handle
mstruct.liargs = {xx,yy};    % arguments for log-likelihood
mstruct.priargs = {};        % extra arguments for prior (besides theta)

% Set optimization parameters for fminunc
opts = optimoptions('fminunc','algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective','display','off');

% intial guess for weights (random)
w0 = randn(nw,1)*.1;

% Compute MAP estimate given this value of theta
lfunc = @(w)(neglogpost_GLM(w,theta0,mstruct));
wmap0 = fminunc(lfunc,w0,opts);  % get MAP estimate


%% 3. Compute updated wmap using DLA and using Newton update

% ======================================================
% Set up Laplace Approximation at wmap0 
% ======================================================

% Compute gradient and Hessian of negative log-likelihood at wmap0
[~,dnegL0,ddnL0] = mstruct.neglogli(wmap0,mstruct.liargs{:}); 
% Compute Hessian of log-prior at wmap0
[~,~,negCinv] = mstruct.logprior(wmap0,theta0,mstruct.priargs{:});
% Compute posterior Hessian at wmap0
postHess0 = ddnL0-negCinv; % posterior Hessian

% Compute DLA term mean term (Hessian of NLL times log-likelihood mean)
ddnLmu0 = postHess0*wmap0;

% ======================================================
% Compute prior and posterior Hessian at new hyperparameters theta1
% ======================================================

% make new inverse prior covariance
Cinv_giventheta = (1/theta1)*eye(nw);  % inverse prior covariance
    
% Compute updated posterior Hessian
Hess_giventheta = (ddnL0+Cinv_giventheta);

% ======================================================
% Update wmap using DLA 
% ======================================================

% Compute updated w_MAP
wmap_new_DLA = Hess_giventheta\ddnLmu0;


% ======================================================
% Update wmap using 1 Newton step
% ======================================================

gradTerm = -dnegL0 - 1/theta1*wmap0; % gradient 
wmap_new_Newton = wmap0 + Hess_giventheta\gradTerm; % newton update

% ======================================================
% Plot differences
% ======================================================

plot(1:nw, wmap_new_DLA,1:nw, wmap_new_Newton, '--');
xlabel('weight #'); ylabel('weight'); title('updated W_{map}'); box off;
fprintf('max abs diff: %.3f\n',max(abs(wmap_new_DLA-wmap_new_Newton)));

