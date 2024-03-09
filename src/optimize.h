#ifndef OPTIMIZE_H
#define OPTIMIZE_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif /* OPTIMIZE_H */


int opt_D(double ** S, double ** Theta, 
        double * D, int p, int K, int * n, double lr, double tol, int maxit){
    int val = 1;
    double val2 = -1;
    double lr_decay = 0.9, lr_decay_c = 1-lr_decay, eps = 0.00000001;
    double f_new, f_old;
    int i,j,h;
    double * cache = calloc(p,sizeof(double));    
    double * temp_grad = calloc(p,sizeof(double));    
    lg dlg_ins = loss_grad_D(S,D,Theta,p,K,n);    
    f_old = dlg_ins.loss_val;
    for(i = 0; i < maxit; i++){
        R_CheckUserInterrupt();
        F77_NAME(dcopy)(&p,dlg_ins.grad,&val,temp_grad,&val);
        for(j = 0; j < p; j++){
            temp_grad[j] *= temp_grad[j];
        }
        F77_NAME(dscal)(&p,&lr_decay,cache,&val);
        F77_NAME(daxpy)(&p,&lr_decay_c,temp_grad,&val,cache,&val);   
        for(j = 0; j < p; j++){
            dlg_ins.grad[j] /= sqrt(cache[j])+eps;
        }
        F77_NAME(dscal)(&p,&lr,dlg_ins.grad,&val);
        for(h = 0; h < K; h++){
            for(j = 0; j < p; j++){
                *(Theta[h]+(p+1)*j) -= dlg_ins.grad[j];
            }
        }
        F77_NAME(daxpy)(&p,&val2,dlg_ins.grad,&val,D,&val);  
        free(dlg_ins.grad);
        dlg_ins = loss_grad_D(S,D,Theta,p,K,n);
        f_new = dlg_ins.loss_val;
        /*if(f_new > f_old){
            lr /= 2;
            printf("D: Automatically update lr = %f \n",lr);
        }*/
        if(fabs(f_new - f_old)<tol){
            break;
        }
        f_old = f_new;
    }
    free(cache);
    free(temp_grad);
    free(dlg_ins.grad);
    return 0;    
}

int opt_L(double ** S, double ** Theta, double * L,
        double * R, int p, int r, int K, int * n, double * lr, double tol, int maxit){    
    if(r == 0){
        return 0;
    }
    int val = 1, val1 = p*r, val5 = p*p;
    double val0 = 1, val2 = 0.5, val3 = -1, val4 = 0;
    const char CN = 'N', CT = 'T';
    double lr_decay = 0.9, lr_decay_c = 1-lr_decay, eps = 0.00000001;
    double f_new, f_old;
    int i,j;
    int restore = 0;
    double * cache = calloc(p*r,sizeof(double));    
    double * temp_grad = calloc(p*r,sizeof(double));    
    double * Delta_L = calloc(p*p,sizeof(double));
    lg lg_ins = loss_grad_L(S,Theta,R,p,r,K,n);    
    f_old = lg_ins.loss_val;
    for(i = 0; i < maxit; i++){
        R_CheckUserInterrupt();
        if(restore == 0){
            F77_NAME(dcopy)(&val1,lg_ins.grad,&val,temp_grad,&val);
            for(j = 0; j < p*r; j++){
                temp_grad[j] *= temp_grad[j];
            }
            F77_NAME(dscal)(&val1,&lr_decay,cache,&val);
            F77_NAME(daxpy)(&val1,&lr_decay_c,temp_grad,&val,cache,&val);                       
            for(j = 0; j < p*r; j++){
                lg_ins.grad[j] /= sqrt(cache[j])+eps;
            }
            F77_NAME(dscal)(&val1,lr,lg_ins.grad,&val);
        }else{
            F77_NAME(dscal)(&val1,&val2,lg_ins.grad,&val);
            restore = 0;
        }
        F77_NAME(daxpy)(&val1,&val3,lg_ins.grad,&val,R,&val);
        F77_NAME(dgemm)(&CN,&CT,&p,&p,&r,&val0,R,&p,R,&p,&val4,Delta_L,&p); 
        F77_NAME(daxpy)(&val5,&val3,L,&val,Delta_L,&val);   
        for(j = 0; j < K; j++){
            F77_NAME(daxpy)(&val5,&val3,Delta_L,&val,Theta[j],&val);        
        }
        F77_NAME(dcopy)(&val1,lg_ins.grad,&val,temp_grad,&val);
        free(lg_ins.grad);
        lg_ins = loss_grad_L(S,Theta,R,p,r,K,n);
        f_new = lg_ins.loss_val;
        if(f_new == -10000000){            
            (*lr) /= 2.0;
            Rprintf("L (not positive definite): Automatically update lr = %f \n",(*lr));
            F77_NAME(daxpy)(&val1,&val0,temp_grad,&val,R,&val);
            for(j = 0; j < K; j++){
                F77_NAME(daxpy)(&val5,&val0,Delta_L,&val,Theta[j],&val);        
            }
            lg_ins.grad = calloc(p*r,sizeof(double));
            F77_NAME(dcopy)(&val1,temp_grad,&val,lg_ins.grad,&val);
            restore = 1;
            continue;
        }
        F77_NAME(daxpy)(&val5,&val0,Delta_L,&val,L,&val);        
        /*if(f_new > f_old){            
            (*lr) /= 1.2;
            printf("L: Automatically update lr = %f \n",(*lr));
        }*/
        if(fabs(f_new - f_old)<tol){
            break;
        }
        f_old = f_new;
    } 
    /*printf("L Step: iter = %d \n",i);*/
    free(cache);
    free(temp_grad);
    free(Delta_L);
    free(lg_ins.grad);
    return 0;    
}

int a_opt_JDL_L(double ** S, double * D, double * L, 
        double * R, int p, int r, int K, int * n){
    if(r == 0){
        return 0;
    }
    double val0 = 1, val1 = 0, val3 = -1;
    int val2 = p-r+1;
    const char CN = 'N', CT = 'T', CV = 'V', CI = 'I', CU = 'U';
    int info, outM = r;
    int lwork = MAX(1,26*p), liwork = MAX(1,10*p);
    double work[lwork];
    int iwork[liwork];
    int i,j;
    double * D_sqrt = calloc(p,sizeof(double));
    for(i = 0; i < p; i++){
        D_sqrt[i] = sqrt(D[i]);
    }
    double * DSD = calloc(p*p,sizeof(double));
    for(i = 0; i < p; i++){        
        for(j = 0; j < p; j++){
            for(int h = 0; h < K; h++){
                *(DSD+p*i+j) += *(S[h]+p*i+j)*((double)n[h])/n[K];
            }
            *(DSD+p*i+j) *= D_sqrt[i]*D_sqrt[j];
        }
    }
    /* Eigendecomposition */   
    int outISUPPZ[2*MAX(1,r)];
    double outW[p]; /* Eigenvalues */
    double outZ[p][MAX(1,r)]; /* Eigenvectors */
    F77_NAME(dsyevr)(&CV,&CI,&CU,&p,DSD,&p,&val1,&val1,&val2,&p,&val3,
            &outM,outW,*outZ,&p,outISUPPZ,work,&lwork,iwork,&liwork,&info);
    double temp;
    for(i = 0; i < r; i++){
        temp = 1-1/fmax(outW[i],1.0);
        for(j = 0; j < p; j++){                
            *(R+i*p+j) = *((*outZ)+i*p+j) * sqrt(temp)*D_sqrt[j];
        }
    }
    F77_NAME(dgemm)(&CN,&CT,&p,&p,&r,&val0,R,&p,R,&p,&val1,L,&p); 
    free(D_sqrt);
    free(DSD);
    return 0;    
}

int a_opt_JDL_Ls(double ** S, double * D, double * L, 
        double ** Ls, int p, int * rks, int K){
    double val0 = 1, val2 = -1, val3 = 0;
    int val = 1, val1 = p*p;
    int val4;
    const char CR = 'R', CL = 'L', CU = 'U', CN = 'N', CT = 'T', CV = 'V', CI = 'I';
    int info, outM;
    int lwork = MAX(1,26*p), liwork = MAX(1,10*p);
    double work[lwork];
    int iwork[liwork];

    int i,j;
    double temp = 0;    
    /* Cholesky instead of Eigendecomposition of DmL */
    double * DmL_sqrt = calloc(p*p,sizeof(double));
    F77_NAME(dcopy)(&val1,L,&val,DmL_sqrt,&val);
    F77_NAME(dscal)(&val1,&val2,DmL_sqrt,&val);
    for(i = 0; i < p; i++){
        *(DmL_sqrt+i*(p+1)) += D[i];
    }
    F77_NAME(dpotrf)(&CU, &p, DmL_sqrt, &p, &info);
    /* DmL_sqrt ready */
    for(i = 0; i < K; i ++){
        if(rks[i] == 0){
            continue;
        }   
        /* Create DSD */        
        double * DSD = calloc(p*p,sizeof(double));
        memcpy(DSD,S[i],sizeof(double)*p*p);
        F77_NAME(dtrmm)(&CL,&CU,&CN,&CN,&p,&p,&val0,DmL_sqrt,&p,DSD,&p);
        F77_NAME(dtrmm)(&CR,&CU,&CT,&CN,&p,&p,&val0,DmL_sqrt,&p,DSD,&p);  
        /* Eigendecomposition */
        int outISUPPZ[2*MAX(1,rks[i])];
        double outW[p]; /* Eigenvalues */
        double outZ[p][MAX(1,rks[i])]; /* Eigenvectors */
        val4 = p-rks[i]+1;
        F77_NAME(dsyevr)(&CV,&CI,&CU,&p,DSD,&p,&val3,&val3,&val4,&p,&val3,
            &outM,outW,*outZ,&p,outISUPPZ,work,&lwork,iwork,&liwork,&info);
        /* Eigenvalues ready */
        F77_NAME(dtrmm)(&CL,&CU,&CT,&CN,&p,&rks[i],&val0,DmL_sqrt,&p,*outZ,&p);          
        for(j = 0; j < rks[i]; j ++){  
            temp = sqrt(1-1/fmax(outW[j],1.0));
            F77_NAME(dscal)(&p,&temp,*outZ+j*p,&val);
        }   
        F77_NAME(dgemm)(&CN,&CT,&p,&p,&rks[i],&val0,*outZ,&p,*outZ,&p,&val3,Ls[i],&p); 
        free(DSD);
    }    
    free(DmL_sqrt);
    return 0;    
}
