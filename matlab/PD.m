function SIGMAPD = PD(SIGMA)
[V,D] = eig(SIGMA);      
d= diag(D);            
d(d <= 1e-7) = 1e-7;  
D_c = diag(d);        
SIGMAPD = V*D_c*V';   
end