# Decision tree from scratch 

rm(list = ls())

##########################################################  DATA
# Salary, years played and number of hits of baseball players.

library(uuml)
data("Hitters")

Hitters <- Hitters[complete.cases(Hitters),] # Remove NA
X_test <- Hitters[1:30, c("Years", "Hits")]
y_test <- Hitters[1:30, c("Salary")]
X_train <- Hitters[31:nrow(Hitters), c("Years", "Hits")]
y_train <- Hitters[31:nrow(Hitters), c("Salary")]

##########################################################  SPLIT TREE

tree_split <- function(X, y, l){
  min_k=0
  min_j=0
  min_temp=0
  
  for(j in 1:ncol(X)){   
    for(k in 1:nrow(X)){  
      
      s <- X[k,j] # Split point
      R1 <- which(X[,j] <= s)
      R2 <- which(X[,j] > s)
      y1 = y[which(X[,j] <= s)]
      y2 = y[which(X[,j] > s)]
      
      # If R1 or R2 is smaller than the leaf size l  
      if (length(R1) < l || length(R2) <l){
        next
      }
      
      # Compute c1 and c2
      c1 = 1/length(R1) * sum(y1) # mean of y within Rm 
      c2 = 1/length(R2) * sum(y2)
      
      # Calculate SS
      SS1min = sum((y1-c1)^2)
      SS2min = sum((y2-c2)^2)
      SS=SS1min+SS2min
      
      # Save smallest SS
      if (min_temp==0){
        min_temp=SS
        min_k=k
        min_j=j
      }
      
      if (min_temp>SS){
        min_temp=SS
        min_k=k
        min_j=j 
      }
    }
  }
  # The final results for the min SS
  s = X[min_k, min_j]
  R1 <- which(X[,min_j] <= s)
  R2 <- which(X[,min_j] > s)
  results=list(j = min_j, s = s, R1 = R1, R2 = R2, SS=min_temp)
  return(results)
}

########################################################## GROW TREE

grow_tree <- function(X, y, l){
  #Initial values  
  m=1
  M=1
  init <- tree_split(X, y, l)
  
  temp_r<- list()
  temp_r<- c(temp_r, list(init))
  temp_data <- list()
  temp_data= list(c(temp_data, list(X,y)))
  
  results=as.data.frame(matrix(NA,1,5))
  names(results) <- c("j", "s", "R1_i","R2_i","gamma")
  
  while (m<=M){
    
    temp_x = temp_data[[m]][[1]] # selects x data
    temp_y = temp_data[[m]][[2]] # selects y data
    
    lenght_of_temp_x = nrow(temp_x) 
    
    if(lenght_of_temp_x >= 2*l){
      
      temp_r[[m]] = tree_split(temp_x, temp_y, l)    
      R <- list(temp_r[[m]]$R1, temp_r[[m]]$R2)
      
      x_1 = temp_x[c(temp_r[[m]]$R1),]
      y_1 = temp_y[c(temp_r[[m]]$R1)]
      
      x_2 = temp_x[c(temp_r[[m]]$R2),]
      y_2 = temp_y[c(temp_r[[m]]$R2)]
      
      temp_data[[M+1]] = list(x_1,y_1)
      temp_data[[M+2]] = list(x_2,y_2)
      
      results[m,]<-c(j=temp_r[[m]][1],    
                     s=temp_r[[m]][2],    
                     R1_i=M+1,                         
                     R2_i=M+2,
                     gamma="NA")
      M=M+2
    } else {
      # When region too small to split 
      y_terminal= temp_data[[m]][2]
      gamma= 1/lenght_of_temp_x * sum(y_terminal[[1]])
      
      results[m,]<-c(j="NA",    
                     s="NA",    
                     R1_i="NA",                         
                     R2_i="NA",
                     gamma=gamma)
    }
    m=m+1
  }
  return(results)
}

# Call function 
tree = grow_tree(X_train, y_train, l=5)

########################################################## PREDICTION

predict_with_tree <- function(new_data, tree){
  gamma=matrix(NA, nrow(new_data), 1)
  rownames(gamma) = rownames(new_data)
  colnames(gamma) = "Predicted salary"
  
  for (z in 1:nrow(new_data)){
    R=1
    for(i in 1:nrow(tree)){
      i=R
      if (tree[i,1]=="NA"){
        gamma[z,]= round(as.numeric(tree[R,5]), digits=3)
      }else{
        if (new_data[z,as.numeric(tree[i,1])] <= as.numeric(tree[i,2])){
          R = as.numeric(tree[i,3])
        }else{
          R = as.numeric(tree[i,4])
        }
      }
    }
  }
  return(gamma)
}

# Call function
pred_salary <- predict_with_tree(new_data=X_test, tree=tree)
pred_salary
