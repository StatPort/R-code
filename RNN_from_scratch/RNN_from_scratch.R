# RNN - Implementation of a one-layer recurrent neural network from scratch

rm(list = ls()) 

########################################################## DATA
library(uuml)
data("rnn_example")
X <- rnn_example$embeddings[1:3,,drop=FALSE] # 3 words

################################################ Compute a_t

rnn_unit <- function(h_t_minus_one, X, W, U, b){
  x_t = t(X) 
  a_t = b + W %*% h_t_minus_one + U %*% x_t 
  return(a_t)
}

################################################ Tanh Activation function 

activation <- function(a_t){
  ht = (exp(a_t) - exp(-a_t))/(exp(a_t)+exp(-a_t))  
  return(ht) 
}

################################################ Output

output_rnn <- function(h_t, V, c){
  out = V %*% h_t +c
  return(out)
}

################################################ Softmax

softmax <- function(x){
  softm <- exp(x)/sum(exp(x))
  return(softm)
}

################################################ All together

rnn_layer <- function(X, W, V, U, b, c){
  
  # Initial values  
  h_t_minus_one <- matrix(0, nrow = nrow(W), ncol = 1) # previous ht
  h_t = matrix(NA, nrow(W), nrow(X))  
  output = list() 
  
  for (i in 1:nrow(X)){
    
    # a_t 
    a_t <- rnn_unit(h_t_minus_one, t(X[i,]), W, U, b) 
    
    # ht
    output$h_t[[i]] = t(as.matrix(activation(a_t)))
    
    # yhat
    ht = as.matrix(activation(a_t))
    out = output_rnn(ht, V, c)
    output$yhat[[i]] = t(softmax(out))
    h_t_minus_one = as.matrix(activation(a_t)) # save h_t as h_t_minus_one
  }
  return(output)
}

# Call function 
yhat <- rnn_layer(X, W = rnn_example$W,V = rnn_example$V,U = rnn_example$U,rnn_example$b,rnn_example$c)
yhat


 
