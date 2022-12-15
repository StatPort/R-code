# Transformers and Attention 
# Implementation of a multi-head attention transformer layer from scratch

rm(list = ls()) 

#############################################  DATA 
library(uuml)
data("transformer_example")

# Query, key and value for first attention head
Wq <- transformer_example$Wq[,,1] # query
Wk <- transformer_example$Wk[,,1] # key
Wv <- transformer_example$Wv[,,1] # value

# First 3 words and their embeddings
X <- transformer_example$embeddings[1:3,]

#############################################  COMPUTE QUERY, KEY, VALUE

qkv <- function(X, Wq, Wk, Wv){
  output=list()
  
  for (i in 1:dim(Wq)[3]){
    output$Q[[i]] = X %*% Wq[,,i] 
    output$K[[i]] = X %*% Wk[,,i] 
    output$V[[i]] = X %*% Wv[,,i] 
  }
  return(output)
}

#############################################  COMPUTE ATTENTION
attention <- function(Q, K, V){
  Z = list()
  for (i in 1:length(V)){
    
    #score
    score= Q[[i]] %*% t(K[[i]])
    
    # max for each row in score matrix
    max_val = as.matrix(apply(score, MARGIN=1, FUN=max))
    attention = matrix(0, nrow=nrow(score), ncol=ncol(score))
    
    # Compute attention/weights by softmax
    for (j in 1:nrow(score)) {  
      attention[j,] = exp((score[j,]-max_val[j,]) / sqrt(ncol(K[[i]])))/
        sum(exp((score[j,]-max_val[j,]) / sqrt(ncol(K[[i]]))))
    }
    Z[[i]] = attention %*% V[[i]]
  }
  return(Z)
}

#############################################  ALL TOGETHER

multi_head_self_attention = function(X, Wq, Wk, Wv, W0){
  
  res <- qkv(X, Wq, Wk, Wv)                 # Produce Q, K , V 
  Z_multi = attention(res$Q, res$K, res$V)  # Produce Z for each head 
  
  # Z matrix
  Z_matrix_old = NA
  for (i in 1:length(res$V)){
    Z_matrix_new = matrix(unlist(Z_multi[[i]]), ncol = ncol(Wq), byrow = FALSE)
    Z_matrix_old = rbind(Z_matrix_old,Z_matrix_new)
  }
  Z_matrix = Z_matrix_old[2:nrow(Z_matrix_old),]
  
  # emb out
  out=0
  row=0
  col=nrow(X)
  
  for (i in 1:length(res$V)){
    out_new = Z_matrix[(row+1):(col*i),1:ncol(Z_matrix)] %*% W0[(row+1):(col*i),1:ncol(Z_matrix)]
    out = out_new+out
    row=col*i
  }
  return(out)
}

# Call function
multi_head_self_attention(X, transformer_example$Wq,
                          transformer_example$Wk,
                          transformer_example$Wv,
                          transformer_example$W0)




