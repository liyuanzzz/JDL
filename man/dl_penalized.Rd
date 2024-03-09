\name{dl_penalized}
\alias{dl_penalized}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Precision matrix estimation by diagonal and low-rank decomposition with rank penalty.
}
\description{
	dl_penalized obtains a precision matrix estimator by diagonal and low-rank decomposition with rank penalty.
}
\usage{
dl_penalized(X, rs, lams, Xval = NULL, nfolds = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{
%%     ~~Describe \code{X} here~~
The data matrix.
}
  \item{rs}{
Potential ranks arranged in ascending order.
}
  \item{lams}{
%%     ~~Describe \code{lams} here~~
The tuning parameter set arranged in ascending order.
}
  \item{Xval}{
  A standalone validating set in the form of a list of data matrices.
  }
  \item{nfolds}{
%%     ~~Describe \code{nfolds} here~~
Number of folds for cross-validation.
}
  \item{tol}{
%%     ~~Describe \code{tol} here~~
Tolerance for the convergence of the main objective function for calling dl_fix.
}
  \item{lr_D}{
%%     ~~Describe \code{lr_D} here~~
Learning rate for D step for calling dl_fix.
}
  \item{tol_D}{
%%     ~~Describe \code{tol_D} here~~
Tolerance for the convergence of D step for calling dl_fix.
}
  \item{maxit}{
%%     ~~Describe \code{maxit} here~~
Maximum iterations for both the main objective function and D step for calling dl_fix.
}

}
\details{
If Xval != NULL, the tuning parameter is selected via minimizing the negative log-likelihood of normal distribution on the separate validation set.

If Xval = NULL and nfolds != NULL, the tuning parameter is selected via cross-validation. During each of the nfolds rounds of cross-validation, the data matrices is partitioned into a validating data matrix (with size (sample size)\%/\%nfolds) and a training data matrix; the estimations are performed on the training set and evaluated on the validating set. The tuning parameter is selected via minimizing the average (over nfolds rounds) negative log-likelihood of normal distribution on the validation sets.

If Xval = NULL and nfolds = NULL, dl_penalized takes the first element in lams as the tuning parameter. 

Use dl_fix if other types of cross-validation are preferred. Use dl_fix with initialization for D to obtain solutions for various r sequentially.
}
\value{
\item{Theta}{The estimated precision matrix}
\item{D}{The estimated diagonal component}
\item{L}{The estimated low-rank component}
\item{evals}{The eigenvalues of the low-rank component}
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
  # Generate the random sample
  library(mvtnorm)
  p = 50
  n = 100
  Sigma = 0.8*diag(p)+matrix(rep(0.2,p*p),p,p)
  set.seed(1234)
  X = rmvnorm(n, mean = rep(0,p),Sigma)
  Xval = rmvnorm(n, mean = rep(0,p),Sigma)
  
  # Estimation
  rs = c(1,3,5,7,9)
  lams = seq(0.6,1.4,0.2)    
  result1 = dl_penalized(X, rs = rs, lams = lams, nfolds = 2)  # 2-fold cross-validation
  result2 = dl_penalized(X, rs = rs, lams = lams, Xval = Xval) # Standalone validating set
  
  # Evaluation
  A = Sigma\%*\%result1$Theta
  KL = sum(diag(A))-log(det(A))-p
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
