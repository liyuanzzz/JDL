dl_fix <- function(X, r, D=NULL, tol = 0.01, lr_D = 0.0002, tol_D = 0.0001, maxit = 10000){
  # If D is a warm start, S must have been calculated already, input S instead of X
  if(r < 0 || tol < 0 || lr_D < 0 || tol_D < 0 || maxit < 0){
    stop("dl_fix: invalid input.")
  }
  if(is.null(D)){
    S = cov(X)
    D = diag(S)
    D[D <= 0] = min(D[D > 0])
    D = 1/D
  }else{
    S = X
  }
  p = dim(X)[2]
  Theta = L = array(rep(0,p*p),dim = c(p,p)) 
  evals = rep(0,p)
  ml = 0
  result = .C("opt_DL_R",as.numeric(D),as.numeric(L),as.numeric(Theta),as.numeric(evals),as.double(ml),as.numeric(S),
              as.integer(p),as.integer(r),as.double(tol),as.double(lr_D),as.double(tol_D),as.integer(maxit))
  Theta = array(result[[3]],dim = c(p,p))
  D = result[[1]]
  return(list(Theta = Theta,
              D = D,
              L = diag(D)-Theta,
              evals = result[[4]],
              ml = result[[5]]))
}

jdl_fix <- function(X, rks, D = NULL, n = NULL, tol = 0.01,lr_D = 0.0002,tol_D = 0.0001,lr_L = NULL,tol_L = 0.001,maxit = 10000){
  # If D is a warm start, S must have been calculated already, input S instead of X
  # Also input sample size n in this case
  if(sum(rks < 0)>0 || tol < 0 || lr_D < 0 || tol_D < 0 || tol_L < 0 || maxit < 0){
    stop("jdl_fix: invalid input.")
  }
  if(is.null(D)){
    K = length(X)
    p = dim(X[[1]])[2]
    n = rep(0,K+1)
    for(k in 1:K){
      n[k] = dim(X[[k]])[1]
    }
    n[K+1] = sum(n[1:K])
    D = 0
    S = array(rep(0,p*p*K),dim=c(p,p,K))
    for(i in 1:K){
      S[,,i] = cov(X[[i]])
      D = D + diag(S[,,i])*n[i]
    }
    D[D <= 0] = min(D[D > 0])
    D = n[K+1]/D
  }else{
    if(is.list(X)){
      K = length(X)
      p = dim(X[[1]])[2]
      S = array(rep(0,p*p*K),dim=c(p,p,K))
      for(i in 1:K){
        S[,,i] = X[[i]]
      }
    }else{
      S = X
      dS = dim(S)
      p = dS[1]
      K = dS[3]
    }
    n = c(n,sum(n))
  }
  # set up parameters
  nr = length(rks)
  r = rks[nr]
  if(is.null(lr_L)){
    lr_L = 0.01/p
  }
  rks = rks[-nr]
  # set up input for C function
  Ls = Theta = array(rep(0,p*p*K),dim=c(p,p,K))
  L = array(rep(0,p*p),dim = c(p,p)) 
  evals = rep(0,p)
  ml = 0
  # Call C function for estimation
  result = .C("opt_JDL_R",as.numeric(D),as.numeric(L),as.numeric(Ls),as.numeric(Theta),as.numeric(evals),as.double(ml),as.numeric(S)
              ,as.integer(p),as.integer(rks),as.integer(r),as.integer(K),as.integer(n),as.double(tol),as.double(lr_D),
              as.double(tol_D),as.double(lr_L),as.double(tol_L),as.integer(maxit))
  # Return useful output
  temp_Ls = array(result[[3]],dim=c(p,p,K))
  temp = array(result[[4]],dim=c(p,p,K))
  Ls = list()  
  Theta = list()
  for(i in 1:K){
    Theta[[i]] = temp[,,i]
    Ls[[i]] = temp_Ls[,,i]
  }
  D = result[[1]]
  L = diag(D) - Theta[[1]] - Ls[[1]]
  return(list(Theta = Theta,
              D = D, 
              L = L, 
              Ls = Ls,
              evals = result[[5]], 
              ml = result[[6]]))
}

dl_penalized <- function(X, rs, lams, Xval = NULL, nfolds = NULL, tol = 0.01, lr_D = 0.0002, tol_D = 0.0001, maxit = 10000){
  dims = dim(X)
  n = dims[1]
  p = dims[2]
  pes = (2*p*(rs+1)-rs*(rs-1))
  if(!is.null(Xval) || !is.null(nfolds)){
    # Validation or Cross-validation
    if(!is.null(Xval)){
      print("Validating")
      nfolds = 1
    }else{
      print("Cross-validating...")
      size = n%/%nfolds
    }
    lk = rep(0,length(rs)) # likelihood part for every r
    loss = rep(0,length(rs)) # cv loss for every r
    score = rep(0,length(lams))
    for(i in 1:nfolds){
      if(!is.null(Xval)){
        Y = X
        Z = Xval
      }else{
        print(paste("fold = ",i,sep = ""))
        Te = ((i-1)*size+1):(i*size)
        Y = X[-Te,] # training set
        Z = X[Te,]  # validation set
      }
      ny = dim(Y)[1]
      Sy = cov(Y)
      Sz = cov(Z)
      Dy = diag(Sy)
      Dy[Dy <= 0] = min(Dy[Dy > 0])
      Dy = 1/Dy
      if(!is.null(Xval)){
        Reys = list()
      }
      for(j in 1:length(rs)){
        r = rs[j]
        Rey = dl_fix(Sy,r,Dy,tol = tol,lr_D = lr_D,tol_D = tol_D,maxit = maxit)
        if(!is.null(Xval)){
          Reys[[j]] = Rey
        }
        lk[j] = Rey$ml
        A = Sz%*%(Rey$Theta)
        loss[j] = sum(diag(A))-log(det(Rey$Theta))
        Dy = Rey$D
      }
      for(h in 1:length(lams)){
        score[h] = score[h] + loss[which.min(lk+lams[h]*pes/ny)]
      }
    }
    inds = which(score == min(score))
    ind = inds[ceiling(length(inds)/2)]
    lam = lams[ind]
    if(!is.null(Xval)){
      rind = which.min(lk+lam*pes/ny)
      dlmin = Reys[[rind]]
      dlmin$r = rs[rind]
      dlmin$lambda = lam
      return(dlmin)
    }
  }else{
    # No cross-validation, expecting only one tuning parameter
    lam = lams[1]
  }
  print("Estimating...")
  S = cov(X)
  D = diag(S)
  D[D <= 0] = min(D[D > 0])
  D = 1/D
  pes = lam*pes/n
  dl = NULL
  for(i in 1:length(rs)){
    r = rs[i]
    dl = dl_fix(S,r,D,tol = tol,lr_D = lr_D,tol_D = tol_D,maxit = maxit)
    f = dl$ml + pes[i]
    if(i == 1||f < fmin){
      rmin = r
      fmin = f 
      dlmin = dl
    }
    D = dl$D
  }
  dlmin$r = rmin
  dlmin$lambda = lam
  return(dlmin)
}

jdl_penalized <- function(X,rks,rs,lams,Xval = NULL,nfolds = NULL,
                          tol = 0.01,lr_D = 0.0002,tol_D = 0.0001,lr_L = NULL,tol_L = 0.001,maxit = 10000){
  rs = rs[rs <= min(rks)]
  K = length(X)
  p = dim(X[[1]])[2]
  n = unlist(lapply(X,nrow))
  pes = 2*p*(rs+1)-rs*(rs-1)
  for(k in 1:K){
    rk = rks[k]
    pes = pes + (rk-rs)*(2*p-rk+rs+1)
  }
  if(!is.null(Xval) || !is.null(nfolds)){
    # Validation or Cross-validation
    if(!is.null(Xval)){
      print("Validating...")
      nfolds = 1
      ny = as.numeric(lapply(Xval,nrow))
    }else{
      print("Cross-validating...")
      size = n%/%nfolds
      ny = rep(0,K)
    }
    lk = rep(0,length(rs)) # likelihood part for every r
    score = rep(0,length(lams))
    Sy = Sz = array(rep(0,p*p*K),dim = c(p,p,K))
    for(i in 1:nfolds){
      if(!is.null(Xval)){
        Dy = 0
        for(k in 1:K){
          Sy[,,k] = cov(X[[k]])
          Sz[,,k] = cov(Xval[[k]])
          Dy = Dy + diag(Sy[,,k])*ny[k]
        }
      }else{
        print(paste("fold = ",i,sep = ""))
        Dy = 0
        for(k in 1:K){
          Te = ((i-1)*size[k]+1):(i*size[k])
          Y = X[[k]][-Te,] # training set
          ny[k] = dim(Y)[1]
          Z = X[[k]][Te,]  # validation set
          Sy[,,k] = cov(Y)
          Sz[,,k] = cov(Z)
          Dy = Dy + diag(Sy[,,k])*ny[k]
        }
      }
      Dy[Dy <= 0] = min(Dy[Dy > 0])
      Dy = sum(ny)/Dy
      loss = rep(0,length(rs)) # cv loss for every r
      if(!is.null(Xval)){
        Reys = list()
      }
      for(j in 1:length(rs)){
        r = rs[j]
        Rey = jdl_fix(Sy,rks=c(rks,r),D = Dy, n=ny, tol = tol, 
                      lr_D = lr_D, tol_D = tol_D, lr_L = lr_L, tol_L = tol_L, maxit = maxit)
        if(!is.null(Xval)){
          Reys[[j]] = Rey
        }
        lk[j] = Rey$ml
        for(k in 1:K){
          A = Sz[,,k]%*%(Rey$Theta[[k]])
          loss[j] = loss[j] + (sum(diag(A))-log(det((Rey$Theta)[[k]])))*ny[k]
        }
        Dy = Rey$D
      }
      loss = loss/sum(ny)
      for(h in 1:length(lams)){
        score[h] = score[h] + loss[which.min(lk+lams[h]*pes/sum(ny))]
      }
    }
    inds = which(score == min(score))
    ind = inds[ceiling(length(inds)/2)]
    lam = lams[ind]
    if(!is.null(Xval)){
      rind = which.min(lk+lam*pes/sum(ny))
      jdlmin = Reys[[rind]]
      jdlmin$r = rs[rind]
      jdlmin$lambda = lam
      return(jdlmin)
    }
  }else{
    # No cross-validation, expecting only one tuning parameter
    lam = lams[1]
  }
  print("Estimating...")
  D = 0
  S = array(rep(0,p*p*K),dim=c(p,p,K))
  for(i in 1:K){
    S[,,i] = cov(X[[i]])
    D = D + diag(S[,,i])*n[i]
  }
  D[D <= 0] = min(D[D > 0])
  D = sum(n)/D
  pes = pes*lam
  jdlmin = NULL
  for(i in 1:length(rs)){
    r = rs[i]
    jdl = jdl_fix(S,c(rks,r),D, n = n, tol = tol, 
                  lr_D = lr_D, tol_D = tol_D, lr_L = lr_L, tol_L = tol_L, maxit = maxit)
    f = jdl$ml + pes[i]/sum(n)
    if(i == 1||f < fmin){
      rmin = r
      fmin = f 
      jdlmin = jdl
    }
    D = jdl$D
  }
  jdlmin$r = rmin
  jdlmin$lambda = lam
  return(jdlmin)
}