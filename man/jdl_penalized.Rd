\name{jdl_penalized}
\alias{jdl_penalized}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Precision matrix estimation by joint diagonal and low-rank decomposition with rank penalty.
}
\description{
jdl_penalized obtains a list of precision matrix estimators by joint diagonal and low-rank decomposition with rank penalty.
}
\usage{
jdl_penalized(X, rks, rs, lams, Xval = NULL, nfolds = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{
  A list of data matrices.
}
  \item{rks}{
%%     ~~Describe \code{rks} here~~
A vector of fixed individual ranks.
}
  \item{rs}{
Potential joint ranks arranged in ascending order.
}
  \item{lams}{
The tuning parameter set arranged in ascending order.
}
  \item{Xval}{
  A standalone validating set in the form of a list of data matrices.
  }
  \item{nfolds}{
  Number of folds for cross-validation.
}
  \item{tol}{
Tolerance for the convergence of the main objective function for calling jdl_fix.
}
  \item{lr_D}{
Learning rate for D step for calling jdl_fix.
}
  \item{tol_D}{
Tolerance for the convergence of D step for calling jdl_fix.
}
  \item{lr_L}{
Learning rate for L step (obtaining the shared low-rank component) for calling jdl_fix. If no input, lr_L = 0.01/p.
}
  \item{tol_L}{
Tolerance for the convergence of L step for calling jdl_fix.
}
  \item{maxit}{
Maximum iterations for both the main objective function, D step and L step for calling jdl_fix.
}
}
\details{
If Xval != NULL, the tuning parameter is selected via minimizing the negative log-likelihood of normal distribution on the separate validation set.

If Xval = NULL and nfolds != NULL, the tuning parameter is selected via cross-validation. During each of the nfolds rounds of cross-validation, each of the K data matrices is partitioned into a validating data matrix (with size (sample size)\%/\%nfolds) and a training data matrix; the estimations are performed on the training set and evaluated on the validating set. The tuning parameter is selected via minimizing the average (over nfolds rounds) negative log-likelihood of normal distribution on the validation sets.

If Xval = NULL and nfolds = NULL, jdl_penalized takes the first element in lams as the tuning parameter.

Use jdl_fix if other types of cross-validation are preferred. Use jdl_fix with initialization for D to obtain solutions for various shared ranks sequentially.
}
\value{
\item{Theta}{A list of estimated precision matrices}
\item{D}{The estimated shared diagonal component}
\item{L}{The estimated shared low-rank component}
\item{evals}{The eigenvalues of the shared low-rank component}
\item{ml}{Value of the objective function}
\item{r}{The selected rank}
\item{lambda}{The selected tuning parameter}
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
  # Generate random samples
  library(mvtnorm)
  p = 50
  n = c(200,200)
  K = 2
  Sigma = list()
  X = list()
  Xval = list()
  set.seed(1234)
  for(i in 1:K){
    v = runif(p)
    Sigma[[i]] = 0.5*diag(p)+matrix(rep(0.2,p*p),p,p)+0.2*v\%o\%v
    X[[i]] = rmvnorm(n[i], mean = rep(0,p),Sigma[[i]])
    Xval[[i]] = rmvnorm(n[i], mean = rep(0,p),Sigma[[i]])
  }
  
  # Estimation
  lams = seq(0.8,1.8,0.2)
  result1 = jdl_penalized(X,c(2,2),c(0,1,2),lams = lams,nfolds = 2)  # 2-fold cross-validation
  result2 = jdl_penalized(X,c(2,2),c(0,1,2),lams = lams,Xval = Xval) # Standalone validating set
  
  # Evaluation
  KL = -K*p
  for(i in 1:K){
    A = Sigma[[i]]\%*\%result1$Theta[[i]]
    KL = KL + sum(diag(A))-log(det(A))
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
