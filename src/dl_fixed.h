#ifndef DL_FIXED_H
#define DL_FIXED_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif /* DL_FIXED_H */

int DL_cal_Theta(double * D, double * L, double ** Theta, int p){
    int val = 1, val1 = p*p;
    double val2 = -1;
    F77_NAME(dcopy)(&val1,L,&val,Theta[0],&val);
    F77_NAME(dscal)(&val1,&val2,Theta[0],&val);
    for(int j = 0; j < p; j++){            
            *(Theta[0]+j*(p+1)) += D[j];
    }    
    return 0;
}

int opt_DL(double * D, double * L, double ** Theta,
        double * evals, double * ml, double ** S, int p, int r, 
        double tol, double lr_D, double tol_D, int maxit){
    int n[2] = {1,1};
    if(r < 1){
        DL_cal_Theta(D,L,Theta,p);
        *ml = loss(S,Theta,p,1,n,0,1);  
        return 0;
    }
    int i;
    double f_new, f_old;  
    double * R = calloc(p*r,sizeof(double));
    DL_cal_Theta(D,L,Theta,p);
    f_old = loss(S,Theta,p,1,n,0,1);    
    for(i = 0; i < maxit; i++){  
        R_CheckUserInterrupt();           
        a_opt_JDL_L(S,D,L,R,p,r,1,n);
        DL_cal_Theta(D,L,Theta,p);
        f_new = loss(S,Theta,p,1,n,0,1);
        if(fabs(f_new-f_old)<tol){
            break;
        }
        f_old = f_new;
        R_CheckUserInterrupt();
        opt_D(S,Theta,D,p,1,n,lr_D,tol_D,maxit); 
    }
    free(R);
    Rprintf("****** DL: r = %d, iter = %d ******\n",r,i);
    *ml = f_new;
    double val = 0;
    int val1 = 0;
    const char CN = 'N', CA = 'A', CU = 'U';
    int info,outM;
    int lwork = MAX(1,26*p);
    double work[lwork];
    int liwork = MAX(1,10*p);
    int iwork[liwork];
    int outISUPPZ[2*p];
    double outW[p]; /* Eigenvalues */
    double outZ[p][p]; /* Eigenvectors */
    F77_NAME(dsyevr)(&CN,&CA,&CU,&p,L,&p,&val,&val,&val1,&val1,&val,
        &outM,outW,*outZ,&p,outISUPPZ,work,&lwork,iwork,&liwork,&info);
    for(i = 0; i < p; i++){
        evals[i] = outW[p-i-1];
    }
    /* L deprecated after eigendecomposition*/
    return 0;
}

