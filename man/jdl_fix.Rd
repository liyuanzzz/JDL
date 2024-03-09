\name{jdl_fix}
\alias{jdl_fix}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Precision matrix estimation by joint diagonal and low-rank decomposition with fixed rank.

}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
jdl_fix(X, rks, D = NULL, n = NULL, tol = 0.01, lr_D = 2e-04, tol_D = 0.0001, lr_L = NULL, tol_L = 0.001, maxit = 10000)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{
A list of data matrices if D = NULL or sample covariance matrices if D is initialized.
}
  \item{rks}{
A vector of fixed individual ranks and shared rank (the last element).
}
  \item{D}{
The initialization of the diagonal component. Input the diagonal elements as a vector.
}
  \item{n}{
A vector of sample sizes if D is initialized and X is a list of sample covariance matrices. Not referenced if D = NULL.
}
  \item{tol}{
Tolerance for the convergence of the main objective function.
}
  \item{lr_D}{
Learning rate for D step.
}
  \item{tol_D}{
Tolerance for the convergence of D step.
}
  \item{lr_L}{
Learning rate for L step (obtaining the shared low-rank component). If no input, lr_L = 0.01/p.
}
  \item{tol_L}{
Tolerance for the convergence of L step.
}
  \item{maxit}{
Maximum iterations for both the main objective function, D step and L step.
}
}
\details{
Use jdl_fix with initialization for D to obtain solutions for various shared ranks sequentially.
}
\value{
\item{Theta}{A list of estimated precision matrices}
\item{D}{The estimated shared diagonal component}
\item{L}{The estimated shared low-rank component}
\item{evals}{The eigenvalues of the shared low-rank component}
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
  # Generate random samples
  library(mvtnorm)
  p = 50
  n = c(100,100)
  K = 2
  Sigma = list()
  X = list()
  S = list()
  for(i in 1:K){
    v = runif(p)
    Sigma[[i]] = 0.5*diag(p)+matrix(rep(0.2,p*p),p,p)+0.2*v\%o\%v
    X[[i]] = rmvnorm(n[i], mean = rep(0,p),Sigma[[i]])
    S[[i]] = cov(X[[i]])
  }
  
  # Without initialization for D
  result = jdl_fix(X,c(2,2,1))
  
  # With initialization for D
  D = 0
  for(i in 1:K){
      S[[i]] = cov(X[[i]])
      D = D + diag(S[[i]])*n[i]
    }
  D = sum(n)/D
  result = jdl_fix(S,c(2,2,1),D,n)
  
  # Evaluation
  KL = -K*p
  for(i in 1:K){
    A = Sigma[[i]]\%*\%result$Theta[[i]]
    KL = KL + sum(diag(A))-log(det(A))
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
