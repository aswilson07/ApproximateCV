library(tidyverse)
library(matrixStats)

set.seed(98765)
n=10000
k=3
epsilon=rnorm(n*k) %>% 
        matrix(k,n)
epsilon=sqrt(n)*(epsilon-rowMeans(epsilon))/rowSds(epsilon)
loo_mean_epsilon=-epsilon/n


cv_ridge <- function(lambda, loo_means_x, x, a) {
    kappa = a/ (a + lambda)
    loo_beta = kappa*loo_means_x
    
    return(mean((loo_beta-x)^2)-n+1)
}

cv_lasso <- function(lambda, loo_means_x, x) {
    loo_beta = pmin(pmax(loo_means_x-lambda, 0), loo_means_x+lambda)
    
    return(mean((loo_beta-x)^2)-n+1)
}

#######################

lambda_grid = seq(0, 2.5, .01)

xbar=c(sqrt(1/8), sqrt(9/8), 2)
loo_means_x = xbar + loo_mean_epsilon
x=xbar + epsilon

example_lasso= function(lambda) {
    cv_lasso(lambda, loo_means_x[1,], x[1,]) +
    cv_lasso(lambda, loo_means_x[2,], x[2,]) +
    cv_lasso(lambda, loo_means_x[3,], x[3,])
}
    
    
tibble(lambda=lambda_grid,
       lasso = map_dbl(lambda, example_lasso)) %>% 
    ggplot(aes(x = lambda, y = lasso)) +
    geom_line() +
    labs(
        x = expression(lambda),
        y = expression(ProxACV)
    ) +
    theme_light() 


ggsave("../Figures/multimodal_CV_lasso_small.pdf", width =2, height=1.5)


#######################

lambda_grid = seq(0, 50, .01)


xbar=c(1.3893, 1.5)
loo_means_x = xbar + loo_mean_epsilon[1:2,]
x=xbar + epsilon[1:2,]


example_ridge= function(lambda) {
    cv_ridge(lambda, loo_means_x[1,], x[1,], 1) +
    cv_ridge(lambda, loo_means_x[2,], x[2,], 40)
}


tibble(lambda=lambda_grid,
       ridge = map_dbl(lambda, example_ridge)) %>% 
    ggplot(aes(x = lambda, y = ridge)) +
    geom_line() +
    ylim(3.4,3.75) +
    labs(
        x = expression(lambda),
        y = expression(ACV)
    ) +
    theme_light() 


ggsave("../Figures/multimodal_CV_ridge_small.pdf", width =2, height=1.5)


