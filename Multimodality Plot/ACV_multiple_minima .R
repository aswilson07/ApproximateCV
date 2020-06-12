library(tidyverse)

sure_ridge <- function(lambda, x, a) {
    kappa = a/ (a + lambda)
    (1-kappa) ^ 2 * x ^ 2 + 2 * kappa
}

sure_lasso <- function(lambda, x, sigma) {
    min(abs(x), lambda) ^ 2 + 2 * sigma * (abs(x) >= lambda)
}

#######################

lambda_grid = seq(0, 2.5, .01)

example_lasso= function(lambda) sure_lasso(lambda, sqrt(1/8), 1) +
    sure_lasso(lambda, sqrt(9/8), 1) +
    sure_lasso(lambda, 2, 1)


tibble(lambda=lambda_grid,
       lasso = map_dbl(lambda, example_lasso)) %>% 
    ggplot(aes(x = lambda, y = lasso)) +
    geom_line() +
    labs(
        x = expression(lambda),
        y = expression(ACV^{IJ})#,
        #title = "Multimodality of ACV for the Lasso penalty"#,
        # caption= paste(c("For quadratic error loss, sample means equal to (", 
        #                  paste(format(c(sqrt(1/8), sqrt(9/8), 2), digits=3), collapse=","), 
        #                  "),\n and sample variance equal to the identity."),
        #                collapse="")
    ) +
    theme_light() 


ggsave("../Figures/multimodal_ACV_lasso.pdf", width =3, height=2)


#######################

lambda_grid = seq(0, 50, .01)

example_ridge= function(lambda) sure_ridge(lambda, 1.3893, 1) +
    sure_ridge(lambda, 1.5, 40)



tibble(lambda=lambda_grid,
       ridge = map_dbl(lambda, example_ridge)) %>% 
    ggplot(aes(x = lambda, y = ridge)) +
    geom_line() +
    ylim(3.4,3.75) +
    labs(
        x = expression(lambda),
        y = expression(ACV^{IJ})#,
        #title = "Multimodality of ACV for the Ridge penalty"#,
        # caption= paste(c("For quadratic error loss, sample means equal to (", 
        #                  paste(format(c(1.3893,1.5), digits=3), collapse=","), 
        #                  "),\n and ratio of eigenvalues equal to 80."),
        #                collapse="")
    ) +
    theme_light() 


ggsave("../Figures/multimodal_ACV_ridge.pdf", width =3, height=2)

#######################

lambda_grid = seq(0, 20, .01)

tibble(lambda=lambda_grid,
       ridge1 = map_dbl(lambda, function(lambda) sure_ridge(lambda, .98, 1) ),
       ridge2 = map_dbl(lambda, function(lambda) sure_ridge(lambda, 1.02, 1) ) ) %>% 
    ggplot(aes(x = lambda)) +
    geom_line(aes(y=ridge1)) +
    geom_line(aes(y=ridge2), color="grey") +
    scale_y_continuous(limits = c(.95,1.4),
                       sec.axis = dup_axis(name=NULL, breaks=c(sure_ridge(20, .98, 1), sure_ridge(20, 1.02, 1)), 
                                           labels=c("x=0.98", "x=1.02"))) +
    labs(
        x = expression(lambda),
        y = expression(ACV^{IJ})#,
        #title = "Flatness of ACV for the Ridge penalty"
    ) +
    theme_light() 


ggsave("../Figures/flat_ACV_ridge.pdf", width =3, height=2)

