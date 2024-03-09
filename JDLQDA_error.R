library(JDL)
library(matrixcalc)
library(MASS)
library(flare)
library(linprog)
library(pracma)

#setting
p=200
N=400
s=10
mu_min=1
pie=0.5
rep=48
error <- rep(0,rep)
error_QDA <- rep(0,rep)

mu1 <- read.table("~/Desktop/block_200/mu1_block_200", quote="\"")
mu1 <- matrix(unlist(mu1), nrow = N/2, byrow = TRUE)
mu2 <- read.table("~/Desktop/block_200/mu2_block_200", quote="\"")
mu2 <- matrix(unlist(mu2), nrow = N/2, byrow = TRUE)
Sigma1 <- read.table("~/Desktop/block_200/Sigma1_block_200", quote="\"")
Sigma1 <- matrix(unlist(Sigma1), nrow = N/2, byrow = TRUE)
Sigma2<- read.table("~/Desktop/block_200/Sigma2_block_200", quote="\"")
Sigma2 <- matrix(unlist(Sigma2), nrow = N/2, byrow = TRUE)

for (w in 1:rep){
  
  #generate
  xt <- mvrnorm(N/2,mu1,Sigma1)
  yt <- mvrnorm(N/2,mu2,Sigma2)
  beta <- inv(Sigma2)%*%(mu2-mu1)
  #estimate
  hatmux <- (colMeans(xt))
  hatmuy <- (colMeans(yt))
  hatdelta <- hatmuy-hatmux
  hatSigmaX <- cov(xt)
  hatSigmaY <- cov(yt)
  hatSigmaX <- hatSigmaX + sqrt(log(p)/N)*diag(1,p,p)
  hatSigmaY <- hatSigmaY + sqrt(log(p)/N)*diag(1,p,p)
  
  #JDL
  lams = seq(0.8,1.8,0.2)
  X = list()
  X[[1]] <- xt
  X[[2]] <- yt
  result1 = jdl_penalized(X,c(2,2),c(0,1,2),lams = lams,nfolds = 2)
  JDLhatSigmaX <- result1$Theta[[1]]
  JDLhatSigmaY <- result1$Theta[[2]]
  #KL = -K*p
  #for(i in 1:K){
  #  A = Sigma[[i]]%*%result1$Theta[[i]]
  #  KL = KL + sum(diag(A))-log(det(A))
  #}
  #KL
  
  
#   ##estimate beta
#   DiscVec <- function(xt, yt, lambda){
#     n <- length(xt[,1])
#     p <- length(xt[1,])
#     hatmux <- (colMeans(xt))
#     hatmuy <- (colMeans(yt))
#     hatdelta <- hatmuy-hatmux
#     hatSigma <- cov(yt)
#     a <- rep(lambda2,p)
#     f <- rep(1,p*2)
#     hbind1 <- rbind(hatSigma,-hatSigma)
#     hbind2 <- rbind(-hatSigma, hatSigma)
#     CoeffM <- cbind(hbind1, hbind2)
#     dbind1 <- matrix(a+hatdelta,nrow = p)
#     dbind2 <- matrix(a-hatdelta,nrow = p)
#     Coeffb <- rbind(dbind1,dbind2)
#     uv <- solveLP(f,Coeffb,CoeffM,zero=1e-7)$solution
#     uv <- matrix(uv,nrow = 2*p)
#     uv[which(abs(uv)<1e-7)] <- 0 
#     hatbeta <- uv[1:p]-uv[(p+1):(2*p)]
#   }
#   #lambda2 = 0.25*max(max(abs(hatSigmaY%*%beta-hatdelta)))
#   lambda2 = 0.2258
#   hatbeta <- DiscVec(xt,yt,lambda2)
  
  #test
  xtest <- mvrnorm(N/2,mu1,Sigma1)
  ytest <- mvrnorm(N/2,mu2,Sigma2)
  ztest <- rbind(xtest,ytest)
  labelz <- c(rep(1,N/2),rep(2,N/2))
  
  IDX_QDA <- rep(0, N)
  IDX_JDLQDA <- rep(0,N)
  
  for (i in c(1:N)){
    z <- ztest[i,]
    reg <- t(z-hatmux)
    IDX_JDLQDA[i]<- (reg)%*%(inv(JDLhatSigmaY)-inv(JDLhatSigmaX))%*%t(reg)-2*(hatmuy-hatmux)%*%inv(JDLhatSigmaY)%*%(z-hatmux/2-hatmuy/2)-log(det(JDLhatSigmaX))+log(det(JDLhatSigmaY))
    IDX_QDA[i]<- (reg)%*%(inv(hatSigmaY)-inv(hatSigmaX))%*%t(reg)-2*(hatmuy-hatmux)%*%inv(hatSigmaY)%*%(z-hatmux/2-hatmuy/2)-log(det(hatSigmaX))+log(det(hatSigmaY))
  }
  
  IDX_JDLQDA <- (IDX_JDLQDA<=1e-06)+1
  IDX_QDA <- (IDX_QDA<=1e-06)+1
  
  round(sum(abs(IDX_JDLQDA-labelz))/N,4)
  round(sum(abs(IDX_QDA-labelz))/N,4)
  error[w] <- round(sum(abs(IDX_JDLQDA-labelz))/N,5)
  error_QDA[w] <- round(sum(abs(IDX_QDA-labelz))/N,4)
  print(error[w])
  print(error_QDA[w])
  print(w)
  write.csv(error,"error.csv")
  write.csv(error_QDA,"error_QDA.csv")
}
write.csv(error,"error.csv")
write.csv(error_QDA,"error_QDA.csv")
mean(error)
std(error)
mean(error_QDA)
std(error_QDA)
