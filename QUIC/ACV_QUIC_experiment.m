% ACV vs CV using QUIC

%Load the Dataset 
dataset_name = 'Arabidopsis';
S_temp = load(strcat(dataset_name,'.mat'));
train_raw_data = S_temp.data;
raw_data = train_raw_data;
n = size(raw_data,1);
d = size(raw_data ,2);

%choose hyperparameters
lambda = .25;
lambda_values = size(lambda,2);

%construct zero mean dataset
data = raw_data - repmat(sum(raw_data)/n,[n,1]);
S = cov(data);

% choose hyperparameters for QUIC function 
acv_iter = 1;
max_iter = 100;
verbosity = 2;
tolerance = 1e-6; 
tol_values =size(tolerance,2);
total_iter = 10;

%loop total number of iterations
for loop = 1:total_iter
    %loop over tolerance values
    for k = 1:tol_values
        tol = tolerance(k);
        %loop over the lambda values
        for j = 1:lambda_values
            %(re)initialize error variables
            CV_err = 0;
            ACV_err = 0;
            lam = lambda(j);
        
            %optimize objective, get full data estimator 
            [X, W, opt, cputime, iter, dGap] = QUIC('default', S, lam, tol, verbosity, max_iter);
            filename = sprintf('betahat_results-%s-lam%g-index%d-tol%g-run%g.mat',dataset_name,lam,tol,loop);
            save(filename,'lam','X','W','cputime','iter','dGap','loop');
        
        %parallelized loop over LOOCV estimators
            parfor i=1:n
            % Calculate leave-one-out Covariance Matrix 
                X_loo = data(i,:)'*data(i,:);
                S_loo = S - X_loo/n;
    
            % Calculate CV
                [X_CV, W_CV, opt_CV, cputime_CV, iter_CV, dGap_CV] = QUIC('default', S_loo, lam, tol, verbosity, max_iter);
        
            %compute objective
                CV_err_i = ComputeObjective(X_CV, X_loo);
                fprintf('CV time %g, Newton steps %d, iteration %d',cputime_CV, iter_CV, i);

            % save output of CV
                 filename_CV = sprintf('cv_results-%s-lam%g-index%d-tol%g-loop%g.mat',dataset_name,lam,i,tol,loop)
                 parsave(filename_CV, lam,i,CV_err_i,cputime_CV,opt_CV,iter_CV,dGap_CV,loop)


            % Run ACV 
                [X_ACV, W_ACV, opt_ACV, cputime_ACV, iter_ACV, dGap_ACV] = QUIC('default', S_loo, lam, tol, verbosity, acv_iter, X, W);

            %compute objective
                ACV_err_i = ComputeObjective(X_ACV, X_loo);
           
            %save information
                filename = sprintf('acv_results-%s-lam%g-index%d-tol%g-loop%g.mat',dataset_name,lam,i,tol,loop)
                parsave(filename,lam,i,ACV_err_i,cputime_ACV,opt_ACV,iter_ACV,dGap_ACV,loop);
            end
        end
    end
end 

function [error] = ComputeObjective(X,data)
    L = chol(X);
    logdetX = 2*sum(log(diag(L)));
    error =  -logdetX + trace(X*data);
end

function parsave(fname,lam,i,error,cpu_time,opt,iter,dgap,loop)
save(fname,'lam','i','error','cpu_time','opt','iter','dgap','loop')
end