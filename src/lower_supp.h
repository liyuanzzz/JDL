#ifndef LOWER_SUPP_H
#define LOWER_SUPP_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif /* LOWER_SUPP_H */

double ** get_ppm(int n, int p, int K){
    double ** ppm = malloc(K*sizeof(double *));
    for(int i = 0; i < K; i++){
        ppm[i] = calloc(n*p,sizeof(double));
    }
    return ppm;
}

double ** copy_ppm(double ** m, int n, int p, int K){
    double ** mcp = get_ppm(n,p,K);
        for(int i = 0; i < K; i++){
            memcpy(mcp[i],m[i],sizeof(double)*n*p);   
        }
    return mcp;
}

int free_ppm(double ** ppm, int K){
    for(int i = 0; i < K; i++){
        free(ppm[i]);
    }
    free(ppm);
    return 0;
}
