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
rep=50

#labelz <- read.table("~/Desktop/joint_200/labelz_joint.txt", quote="\"")
#xt <- read.table("~/Desktop/joint_200/xt_joint.txt", quote="\"")
#yt <- read.table("~/Desktop/joint_200/yt_joint.txt", quote="\"")
#ztest <- read.table("~/Desktop/joint_200/zt_joint.txt", quote="\"")


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


##estimate beta
DiscVec <- function(xt, yt, lambda){
  n <- length(xt[,1])
  p <- length(xt[1,])
  hatmux <- (colMeans(xt))
  hatmuy <- (colMeans(yt))
  hatdelta <- hatmuy-hatmux
  hatSigma <- cov(yt)
  a <- rep(lambda2,p)
  f <- rep(1,p*2)
  hbind1 <- rbind(hatSigma,-hatSigma)
  hbind2 <- rbind(-hatSigma, hatSigma)
  CoeffM <- cbind(hbind1, hbind2)
  dbind1 <- matrix(a+hatdelta,nrow = p)
  dbind2 <- matrix(a-hatdelta,nrow = p)
  Coeffb <- rbind(dbind1,dbind2)
  uv <- solveLP(f,Coeffb,CoeffM,zero=1e-7)$solution
  uv <- matrix(uv,nrow = 2*p)
  uv[which(abs(uv)<1e-7)] <- 0 
  hatbeta <- uv[1:p]-uv[(p+1):(2*p)]
}
#lambda2 = 0.25*max(max(abs(hatSigmaY%*%beta-hatdelta)))
lambda2 = 0.2258
hatbeta <- DiscVec(xt,yt,lambda2)


IDX_QDA <- rep(0, N)
IDX_JDLQDA <- rep(0,N)

for (i in c(1:N)){
  z <- ztest[i,]
  reg <- t(z-hatmux)
  IDX_JDLQDA[i]<- t(reg)%*%(inv(JDLhatSigmaY)-inv(JDLhatSigmaX))%*%(reg)-2*(hatmuy-hatmux)%*%inv(JDLhatSigmaY)%*%t(z-hatmux/2-hatmuy/2)-log(det(JDLhatSigmaX))+log(det(JDLhatSigmaY))
  IDX_QDA[i]<- t(reg)%*%(inv(hatSigmaY)-inv(hatSigmaX))%*%(reg)-2*(hatmuy-hatmux)%*%inv(hatSigmaY)%*%t(z-hatmux/2-hatmuy/2)-log(det(hatSigmaX))+log(det(hatSigmaY))
}

IDX_JDLQDA <- (IDX_JDLQDA<=1e-06)+1
IDX_QDA <- (IDX_QDA<=1e-06)+1

round(sum(abs(IDX_JDLQDA-labelz))/N,4)
round(sum(abs(IDX_QDA-labelz))/N,4)