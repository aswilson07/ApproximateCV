% This function opens the files and computes the values used in the table

%datasets = ['Lymph','Leukemia','Arabidopsis'];
tol = 1e-6;
lam  = .25;
x = zeros(4,2);
y = zeros(3,4);
%choose dataset to plot (there is more efficient way of doing this if
%plotting all datasets at once)
i = 1;
if i==1
        dataset = 'Lymph';
       [mean_rel_error_Lymph,sd_rel_error_Lymph,CV_iter_Lymph, mean_ACV_error_Lymph, mean_CV_error_Lymph,sd_CV_error_Lymph, sd_ACV_error_Lymph, mean_ACV_time_Lymph, mean_CV_time_Lymph,  sd_ACV_time_Lymph, sd_CV_time_Lymph] = getTable(dataset,lam,tol);
elseif i==2
        dataset = 'Leukemia';
        [mean_rel_error_Leukemia, sd_rel_error_Leukemia, CV_iter_Leukemia, mean_ACV_error_Leukemia, mean_CV_error_Leukemia, sd_CV_error_Leukemia, sd_ACV_error_Leukemia, mean_ACV_time_Leukemia, mean_CV_time_Leukemia,  sd_ACV_time_Leukemia, sd_CV_time_Leukemia] = getTable(dataset,lam,tol);
elseif i==3
        dataset = 'Arabidopsis';
        [mean_rel_error_Ara, sd_rel_error_Ara, CV_iter_Ara, mean_ACV_error_Arabidopsis, mean_CV_error_Arabidopsis, sd_CV_error_Arabidopsis, sd_ACV_error_Arabidopsis, mean_ACV_time_Arabidopsis, mean_CV_time_Arabidopsis, sd_ACV_time_Arabidopsis, sd_CV_time_Arabidopsis] = getTable(dataset,lam,tol);
end


function[mean_rel_error,sd_rel_error,CV_iter, mean_ACV_error, mean_CV_error, sd_CV_error, sd_ACV_error, mean_ACV_time, mean_CV_time, sd_ACV_time, sd_CV_time] = getTable(dataset,lam,tol)
    ACV_error = zeros(1,10);
    CV_error = zeros(1,10);
    ACV_time = zeros(1,10);
    CV_time = zeros(1,10);
    CV_iter = zeros(1,10);
    S_temp = load(strcat(dataset,'.mat'));
    raw_data = S_temp.data;
    n = size(raw_data,1);
    % we ran the experiment 10 times. This collects the information for
    % each run a reports the mean timing and errors and the standard
    % deviations. 
    for loop = 1:10
        for i=1:n
           filename_CV = sprintf('cv_results-%s-lam%g-index%d-tol%g-loop%g.mat',dataset,lam,i,tol,loop);
           CV = load(filename_CV);
           CV_error(loop) = CV_error(loop) + CV.error;
           CV_time(loop) = CV_time(loop) + CV.cpu_time;
           CV_iter(loop) = CV_iter(loop) + CV.iter;
       
       filename_ACV = sprintf('acv_results-%s-lam%g-index%d-tol%g-loop%g.mat',dataset,lam,i,tol,loop);
       ACV = load(filename_ACV);
       ACV_error(loop) = ACV_error(loop) + ACV.error;
       ACV_time(loop) = ACV_time(loop) + ACV.cpu_time;       
        end
    end
    ACV_error = ACV_error/n;
    CV_error = CV_error/n;
    rel_error = zeros(1,10);
    for loop = 1:10
        rel_error(loop) = (CV_error(loop) - ACV_error(loop))/CV_error(loop);
    end
    mean_CV_error = sum(CV_error)/10;
    mean_ACV_error = sum(ACV_error)/10;
    mean_rel_error = sum(rel_error)/10;
    mean_CV_time = sum(CV_time)/10;
    mean_ACV_time = sum(ACV_time)/10;
    %mean_ACV_time = sum(ACV_time)/10;

    sd_CV_error = 0;
    sd_ACV_error = 0;
    sd_CV_time = 0;
    sd_ACV_time = 0;
    sd_rel_error = 0;
    
    for loop = 1:10
        sd_CV_error = sd_CV_error + (CV_error(loop) - mean_CV_error)^2;
        sd_CV_time = sd_CV_time + (CV_time(loop) - mean_CV_time)^2;
        sd_ACV_error = sd_ACV_error + (ACV_error(loop) - mean_ACV_error)^2;
        sd_ACV_time = sd_ACV_time + (ACV_time(loop) - mean_ACV_time)^2;
        sd_rel_error = sd_rel_error + (rel_error(loop) - mean_rel_error)^2;

    end
    sd_CV_error = sd_CV_error/(n-1);
    sd_CV_time = sd_CV_time/(n-1);
    sd_ACV_error = sd_ACV_error/(n-1);
    sd_ACV_time = sd_ACV_time/(n-1);
    sd_rel_error = sd_rel_error/(n-1);
    CV_iter = CV_iter/n;
end
