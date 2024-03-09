#define MAX(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b; })
#include <R.h>
#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <string.h>


#include "BLAS.h"
#include "Lapack.h"

#include "lower_supp.h"
#include "supplement.h"
#include "optimize.h"
#include "jdl_fixed.h"
#include "dl_fixed.h"

void opt_DL_R(double * D, double * L, double * Theta, double * evals, 
        double * ml, double * S, int * p, int * r, double * tol, double * lr_D,
        double * tol_D, int * maxit){
    opt_DL(D,L,&Theta,evals,ml,&S,*p,*r,*tol,*lr_D,*tol_D,*maxit);
}

void opt_JDL_R (double * D, double * L, double * pLs, double * pTheta,
        double * evals, double * ml, double * pS, int * p, int * rks,
        int * r, int * K, int * n, double * tol, double * lr_D, double * tol_D, 
        double * lr_L, double * tol_L, int * maxit){  
    
    double ** S = malloc(sizeof(double*)*(*K));
    double ** Theta = malloc(sizeof(double*)*(*K));
    double ** Ls = malloc(sizeof(double*)*(*K));
    for(int i = 0; i < (*K); i++){
        S[i] = pS+i*(*p)*(*p);
        Theta[i] = pTheta+i*(*p)*(*p);
        Ls[i] = pLs+i*(*p)*(*p);
    }
    opt_JDL(D,L,Ls,Theta,evals,ml,S,*p,rks,*r,*K,n, *tol,*lr_D,*tol_D,*lr_L,*tol_L,*maxit);
    free(S);
    free(Theta);
    free(Ls);
}
