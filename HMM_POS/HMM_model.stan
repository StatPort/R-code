// Model based on suggested models in Stan User’s Guide.  
// HMM MODEL STAN 

data {
  int<lower=1> K; // num categories
  int<lower=1> V; // num words
  int<lower=0> T_; // num supervised items
  int<lower=1> T_unsup; // num unsupervised items
  int <lower=1, upper=V> w[T_] ; // words                   
  int <lower=1, upper=K> z[T_] ; // categories              
  int<lower=1 , upper=V> w_new[T_unsup]; // unsup words
  vector<lower=0>[K] alpha; // transit prior
  vector<lower=0>[V] beta; // emit prior
  
}
parameters {
  simplex[K] theta[K]; // transit probs                     
  simplex[V] phi[K]; // emit probs                          

}
model {
  for (k in 1 : K) {
    theta[k] ~ dirichlet(alpha); // sum of theta=1
  }
  for (k in 1 : K) {
    phi[k] ~ dirichlet(beta); // sum of phi=1
  }
  for (t in 1 : T_) {
    w[t] ~ categorical(phi[z[t]]);        
  }
  for (t in 2 : T_) {
    z[t] ~ categorical(theta[z[t - 1]]); // transition
  }
  
 { 
 //forward algorithm computes log p(w_new|...)
 real acc[K];
 real gamma[T_unsup, K];
 for (k in 1 : K) {
  gamma[1, k] = log(phi[k, w_new[1]]);
 }
 for (t in 2 : T_unsup) {
   for (k in 1 : K) {
     for (j in 1 : K) {
       acc[j] = gamma[t - 1, j] + log(theta[j, k]) + log(phi[k, w_new[t]]);
     }
     gamma[t, k] = log_sum_exp(acc);
   }
  }
  target += log_sum_exp(gamma[T_unsup]);
 }
}
generated quantities {
  vector[T_] log_lik; // vector for log_lik values
  int<lower=1, upper=K> z_star[T_unsup];
  real log_p_z_star;
  
  {
    // Viterbi algorithm
    int back_ptr[T_unsup, K];
    real best_logp[T_unsup, K];
    for (k in 1 : K) {
      best_logp[1, K] = log(phi[k, w_new[1]]);
    }
    for (t in 2 : T_unsup) {
      for (k in 1 : K) {
        best_logp[t, k] = negative_infinity();
        for (j in 1 : K) {
          real logp;
          logp = best_logp[t - 1, j] + log(theta[j, k]) + log(phi[k, w_new[t]]);
          if (logp > best_logp[t, k]) {
            back_ptr[t, k] = j;
            best_logp[t, k] = logp;
          }
        }
      }
    }
    log_p_z_star = max(best_logp[T_unsup]);
    for (k in 1 : K) {
      if (best_logp[T_unsup, k] == log_p_z_star) {
        z_star[T_unsup] = k;
      }
    }
    for (t in 1 : (T_unsup - 1)) {
      z_star[T_unsup - t] = back_ptr[T_unsup - t + 1, z_star[T_unsup - t + 1]];
    }
  }
  // likelihood values 
  for (t in 1:T_){
    log_lik[t]=categorical_lpmf(w[t] | phi[z[t]]);
  }
}
