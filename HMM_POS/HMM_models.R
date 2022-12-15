# HMM MODEL R 

rm(list = ls())
# PACKAGES
library(MCMCpack)
library(LaplacesDemon)
library(readxl)
library(rstan)
library(loo)

########################################################## PREPARE DATA

# DATA
data <- read_excel("NER_new.xlsx")
df <- data[1:98,] #words and pos to build model
df_new <- read_excel("w_new.xlsx")

w = as.integer(as.factor(df$Word))  # words
z = as.integer(as.factor(df$pos_4)) # categories
w_new = as.integer(df_new$w_new)    # observed words in new sentence

# CONSTANTS
K = length(unique(df$pos_4))        # num categories
V = length(unique(df$Word))         # num words
T_ = length(df$Word)                # num instances
T_unsup = length(w_new)             # number unsupervised tokens

# PRIORS 
# Model 1 
alpha1 = rep(0.1,K) #transit prior
beta1 = rep(0.1,V)  #emit prior

# Model 2   
alpha2 = rep(0.8,K) #transit prior
beta2 = rep(0.8,V)  #emit prior

# Model 3   
alpha3 = rep(2,K)  #transit prior
beta3 = rep(2,V)   #emit prior

########################################################## RUN MODELS

# Iterations
iter_n = 3500
warmup_n = 1500

# COMPILE MODEL, EXTRACT DRAWS
# Model 1
set.seed(1234)
fit1 <- stan(file='HMM_model.stan', cores=8,
             data=list(K=K,V=V,T_=T_, T_unsup=T_unsup,w=w,z=z,
                       w_new=w_new, alpha=alpha1,beta=beta1), 
             iter=iter_n, warmup = warmup_n, chains = 4, 
             control = list(adapt_delta=0.98))
draws1 <- rstan::extract(fit1)

# Model 2
set.seed(1234)
fit2 <- stan(file='HMM_model.stan', cores=8,
             data=list(K=K,V=V,T_=T_, T_unsup=T_unsup, w=w,z=z,
                       w_new=w_new, alpha=alpha2,beta=beta2), 
             iter=iter_n, warmup = warmup_n, chains = 4, 
             control = list(adapt_delta=0.98))
draws2 <- rstan::extract(fit2)

# Model 3
set.seed(1234)
fit3 <- stan(file='HMM_model.stan', cores=8,
             data=list(K=K,V=V,T_=T_, T_unsup=T_unsup,w=w,z=z,
                       w_new=w_new, alpha=alpha3,beta=beta3), 
             iter=iter_n, warmup = warmup_n, chains = 4, 
             control = list(adapt_delta=0.98))
draws3 <- rstan::extract(fit3)

########################################################## MODEL EVALUATION

# FUNCTIONS TO BE USED 
# Most common sequence
mcsequence <- function(x, drop = FALSE) { 
  xx <- do.call("paste", c(data.frame(x), sep = "\r"))
  tx <- table(xx)
  mx <- names(tx)[which(tx == max(tx))[1]]
  as.vector(x[match(mx, xx), , drop = drop])
}

# Function to calculate how many POS correctly predicted
correct  <- function(pos_predicted_and_correct){
  N <- nrow(pos_predicted_and_correct)
  count = matrix(NA, nrow(pos_predicted_and_correct), 1)
  for (i in 1:N){
    if (pos_predicted_and_correct[i,1]==pos_predicted_and_correct[i,2]){
      count[i,1] = 1
    }else{
      count[i,1] = 0
    }
    list= list(correct = sum(count),
               total = N,
               percent_correct =sum(count)/N)
  }
  return(list)
}

# EVALUATION - Percentage
correct_pos <- as.integer(df_new$z_new)

# Model 1
predicted_pos_m1 <- mcsequence(draws1$z_star)
pos_predicted_and_correct_m1 <- cbind(predicted_pos_m1,correct_pos)   
correct(pos_predicted_and_correct_m1) #nr correct, total, percentage

# Model 2
predicted_pos_m2 <- mcsequence(draws2$z_star)
pos_predicted_and_correct_m2 <- cbind(predicted_pos_m2,correct_pos)   
correct(pos_predicted_and_correct_m2) #nr correct, total, percentage

# Model 3
predicted_pos_m3 <- mcsequence(draws3$z_star)
pos_predicted_and_correct_m3 <- cbind(predicted_pos_m3,correct_pos)   
correct(pos_predicted_and_correct_m3) #nr correct, total, percentage

# EVALUTATION - Rhat 
max_Rhat1 <- max(monitor(fit1)$Rhat,na.rm=TRUE)
max_Rhat2 <- max(monitor(fit2)$Rhat,na.rm=TRUE)
max_Rhat3 <- max(monitor(fit3)$Rhat,na.rm=TRUE)

# EVALUTATION - n_eff 
min_n_eff1 <- min(monitor(fit1)$n_eff,na.rm=TRUE)
min_n_eff2 <- min(monitor(fit2)$n_eff,na.rm=TRUE)
min_n_eff3 <- min(monitor(fit3)$n_eff,na.rm=TRUE)

# EVALUATION - PSIS loo-CV 
# Model 1
log_lik_1 <- extract_log_lik(fit1, merge_chains = FALSE)
r_eff_1 <- relative_eff(exp(log_lik_1), cores = 8)
loo_1 <- loo(log_lik_1, r_eff = r_eff_1, cores = 8)
                                     
# Model 2
log_lik_2 <- extract_log_lik(fit2, merge_chains = FALSE)
r_eff_2 <- relative_eff(exp(log_lik_2), cores = 8)
loo_2 <- loo(log_lik_2, r_eff = r_eff_2, cores = 8)
                                                    
# Model 3
log_lik_3 <- extract_log_lik(fit3, merge_chains = FALSE)
r_eff_3 <- relative_eff(exp(log_lik_3), cores = 8)
loo_3 <- loo(log_lik_3, r_eff = r_eff_3, cores = 8)

# FINAL COMPARISON
loo_compare(loo_1, loo_2, loo_3)



