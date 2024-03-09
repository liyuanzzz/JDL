#ifndef JDL_FIXED_H
#define JDL_FIXED_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif /* JDL_FIXED_H */

int opt_JDL(double * D, double * L, double ** Ls, double ** Theta,
        double * evals, double * ml, double ** S, int p, int * rks, int r, int K, int * n, 
        double tol, double lr_D, double tol_D, double lr_L, double tol_L,int maxit){
    int i;
    double f_new = 0;
    double f_old = 0;
    int numzero = 0;
    /* Check rank */
    for(i = 0; i < K; i++){
        rks[i] -= r;
        if(rks[i] < 0){
            error("All individual ranks have to be larger than the joint rank.\n");
        }
        if(rks[i] == 0){
            numzero++;
        }
    }
    cal_Theta(D,L,Ls,Theta,p,K);   
    f_old = loss(S,Theta,p,K,n,0,1);
    double * R = calloc(p*MAX(1,r),sizeof(double)); 
    if(r != 0){
        a_opt_JDL_L(S,D,L,R,p,r,K,n);
    }
    cal_Theta(D,L,Ls,Theta,p,K);   
    for(i = 0; i < maxit; i++){  
        if(numzero != K){
            R_CheckUserInterrupt();
            a_opt_JDL_Ls(S,D,L,Ls,p,rks,K);
            cal_Theta(D,L,Ls,Theta,p,K);  
        }
        f_new = loss(S,Theta,p,K,n,0,1);
        if(fabs(f_new-f_old)<tol){
            break;
        }
        f_old = f_new;
        R_CheckUserInterrupt();  
        opt_D(S,Theta,D,p,K,n,lr_D,tol_D,maxit);
        if(r != 0){
            R_CheckUserInterrupt();  
            opt_L(S,Theta,L,R,p,r,K,n,&lr_L,tol_L,maxit);
        }
    }
    free(R);
    Rprintf("****** JDL: r = %d, iter = %d ******\n",r,i);    
    *ml = f_new;
    if(r != 0){
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
    }
    return 0;
}

