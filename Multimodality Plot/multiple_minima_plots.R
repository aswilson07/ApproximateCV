library(tidyverse)
library(furrr)

difference_of_minima=function(y) {
    diff_min=Inf
    min_y=min(y)
    
    for (i in 2:length(y)-1) 
        if ((y[i] <= min(y[i-1], y[i+1])) & (y[i] > min_y)) 
            diff_min = min(diff_min, y[i] - min_y)
    
    if ((y[1] <= y[2]) & (y[1] > min_y)) 
        diff_min = min(diff_min, y[1] - min_y)
    if ((y[length(y)] <= y[length(y)-1]) & (y[length(y)] > min_y)) 
        diff_min = min(diff_min, y[length(y)] - min_y)
    
    return(diff_min)
}


a2 = 50
lambda=c(seq(0,2,by=.001), #grid of values for lambda
         1/seq(.5,.001,by=-.001))
kappa1=1/(1+lambda) #corresponding grid of shrinkage factors
kappa2=a2/(a2+lambda)

SURE_over_lambda = function(X1, X2) {
    X1^2*(kappa1-1)^2 + 2*(kappa1-1) +
        X2^2*(kappa2-1)^2 + 2*(kappa2-1)
}

mm_SURE = function(X1,X2) {
    difference_of_minima(SURE_over_lambda(X1,X2))
}


step=.01
range=4
width= range/step +1

plan(multiprocess)

mm_tibble = tibble(
    X1 = rep(seq(0,range,by=step), width),
    X2 = rep(seq(0,range,by=step), each=width),
    diff_of_min=future_map2_dbl(X1,X2,mm_SURE)
)


mm_tibble %>% 
    mutate(scaled_dom=  (1/log(pmin(diff_of_min, .1))))%>% 
    ggplot(aes(x=X1, y=X2, fill=scaled_dom)) +
        geom_raster() +
        theme_minimal() +
        theme(legend.position="none") 
        

ggsave("manifold_trouble_50.pdf", width=5, height=5)
