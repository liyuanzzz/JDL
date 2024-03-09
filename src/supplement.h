#ifndef SUPPLEMENT_H
#define SUPPLEMENT_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif /* SUPPLEMENT_H */

double loss(double ** S, double ** Theta, int p, int K, int * n,
        int KL, int Keep_Theta){
    double val0 = 1, val1 = 0;
    int val = 1, psq = p*p;
    const char CR = 'R', CU = 'U';
    int i, j, info;
    double L = -p;
    double det;
    for(i = 0 ; i < K; i++){    
        L += F77_NAME(ddot)(&psq,S[i],&val,Theta[i],&val)*((double)n[i])/n[K];
    }
    double * C = calloc(p*p,sizeof(double));
    if(KL == 0){        
        for(i = 0 ; i < K; i++){            
            memcpy(C,Theta[i],sizeof(double)*p*p);
            F77_NAME(dpotrf)(&CU, &p, Theta[i], &p, &info);
            if(info != 0){
                free(C);
                return -10000000;
            }            
            det = 1;
            for(j = 0; j < p; j++){
                det *= *(Theta[i] + (p+1)*j);
            }
            L -= 2*log(det)*((double)n[i])/n[K];
            if(Keep_Theta == 1){
                memcpy(Theta[i],C,sizeof(double)*p*p);
            }
        }
    }else if(KL == 1){
        /* Return KL loss (only for the final accuracy assessment)*/
        for(i = 0 ; i < K; i++){
            F77_NAME(dsymm)(&CR,&CU,&p,&p,&val0,S[i],&p,Theta[i],&p,&val1,C,&p);
            int outIPIV[p];
            F77_NAME(dgetrf)(&p,&p,C,&p,outIPIV,&info);
            det = 1;
            for(j = 0; j < p; j++){
                det *= *(C + (p+1)*j);
            }
            L -= log(det)*((double)n[i])/n[K];
        }                
    }
    free(C); 
    return L;
}

int cal_Theta(double * D, double * L,
        double ** Ls, double ** Theta, int p, int K){
    int psq = p*p, val = 1;
    double val1 = -1;
    for(int i = 0; i < K; i ++){
        F77_NAME(dcopy)(&psq,L,&val,Theta[i],&val);
        F77_NAME(dscal)(&psq,&val1,Theta[i],&val);
        F77_NAME(daxpy)(&psq,&val1,Ls[i],&val,Theta[i],&val);
        for(int j = 0; j < p; j++){            
            *(Theta[i]+j*(p+1)) += D[j];
        }
    }
    return 0;
}

typedef struct {
   double    loss_val;
   double *  grad;
} lg;

lg loss_grad_D(double ** S, double * D, double ** Theta, int p, int K, int * n){
    double det;
    double lval = -p;
    double temp = 0;
    double * G = calloc(p,sizeof(double)); 
    int info;
    const char CU = 'U';
    for(int i = 0; i < K; i++){
        /* loss: trace part */
        temp = 0;
        for(int j = 0; j < p; j++){
            temp += *(S[i]+(p+1)*j)*D[j];
        }
        lval += temp*((double)n[i])/n[K];
        /* loss: determinant part */
        double * C = calloc(p*p,sizeof(double));
        memcpy(C,Theta[i],sizeof(double)*p*p);
        F77_NAME(dpotrf)(&CU,&p,C,&p,&info);
        det = 1;
        for(int j = 0; j < p; j++){
            det *= *(C + (p+1)*j);
        }
        lval -= 2*log(det)*((double)n[i])/n[K];
        /* gradient */
        F77_NAME(dpotri)(&CU,&p,C,&p,&info);
        for(int j = 0; j < p; j++){
            G[j] += (*(S[i]+(p+1)*j)-*(C+(p+1)*j))*((double)n[i])/n[K];
        }        
        free(C);
    }
    lg result;
    result.loss_val = lval;
    result.grad = G;
    return result; 
}

lg loss_grad_L(double ** S, double ** Theta, 
        double * R, int p, int r, int K, int * n){
    int val = p*r, val1 = 1, psq = p*p;
    double val0 = 1;
    double val2;
    const char CL = 'L', CU = 'U';
    int info;
    lg result;
    int i;  
    double ** Theta_inv = copy_ppm(Theta,p,p,K);
    result.loss_val = loss(S, Theta_inv, p, K, n, 0, 0);
    if(result.loss_val == -10000000){
        free_ppm(Theta_inv,K);
        return result;
    }
    double * G = calloc(p*r,sizeof(double));
    double * temp = calloc(p*r,sizeof(double));
    double * temp_S = calloc(p*p,sizeof(double));
    for(i = 0; i < K; i++){
        memcpy(temp,R,sizeof(double)*p*r);
        F77_NAME(dpotrs)(&CU,&p,&r,Theta_inv[i],&p,temp,&p,&info);
        val2 = 2*((double)n[i])/n[K];
        F77_NAME(daxpy)(&val,&val2,temp,&val1,G,&val1);
        F77_NAME(daxpy)(&psq,&val2,S[i],&val1,temp_S,&val1);
    }
    val2 = -1;
    F77_NAME(dsymm)(&CL,&CU,&p,&r,&val2,temp_S,&p,R,&p,&val0,G,&p);    
    result.grad = G;    
    free_ppm(Theta_inv,K);
    free(temp);
    free(temp_S);
    return result; 
}

int Init_D(double * D, double ** S, int p, int K, int * n){
    double temp;
    for(int i = 0; i < p; i++){
        temp = 0;
        for(int j = 0; j < K; j++){
            temp += *(S[j]+(p+1)*i)*((double)n[j])/n[K];
        }
        D[i] = 1/temp;
    }
    return 0;
}
