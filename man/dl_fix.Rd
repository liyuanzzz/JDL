\name{dl_fix}
\alias{dl_fix}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
	Precision matrix estimation by diagonal and low-rank decomposition with fixed rank.
%%  ~~function to do ... ~~
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
	dl_fix obtains a precision matrix estimator by diagonal and low-rank decomposition with fixed rank.
}
\usage{
dl_fix(X, r, D = NULL, tol = 0.01, lr_D = 2e-04, tol_D = 0.0001, maxit = 10000)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{
%%     ~~Describe \code{X} here~~
	The data matrix if D = NULL or the sample covariance matrix if D is initialized.
}
  \item{r}{
%%     ~~Describe \code{r} here~~
The fixed rank.
}
  \item{D}{
%%     ~~Describe \code{D} here~~
The initialization of the diagonal component. Input the diagonal elements as a vector.
}
  \item{tol}{
%%     ~~Describe \code{tol} here~~
Tolerance for the convergence of the main objective function.
}
  \item{lr_D}{
%%     ~~Describe \code{lr_D} here~~
Learning rate for D step.
}
  \item{tol_D}{
%%     ~~Describe \code{tol_D} here~~
Tolerance for the convergence of D step.
}
  \item{maxit}{
%%     ~~Describe \code{maxit} here~~
Maximum iterations for both the main objective function and D step.
}
}
\details{
Use dl_fix with initialization for D to obtain solutions for various r sequentially.
}
\value{
\item{Theta}{The estimated precision matrix}
\item{D}{The estimated diagonal component}
\item{L}{The estimated low-rank component}
\item{evals}{The eigenvalues of the low-rank component}
\item{ml}{Value of the objective function}
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
  X = rmvnorm(n, mean = rep(0,p),Sigma)
  
  # Without initialization for D
  result = dl_fix(X,1)
  
  # With initialization for D
  S = cov(X)
  D = 1/diag(S)
  result = dl_fix(S,1,D)
  
  # Evaluate with KL loss
  A = result$Theta\%*\%Sigma
  KL = sum(diag(A))-log(det(A))-p

}
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line