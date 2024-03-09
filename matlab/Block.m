p=400;
A=[];
U=[];
Omega1=[];
for i=1:p
    A(i,i)=unifrnd(1,2);
    for j=1:p
        U(i,j)=normrnd(0,1);
    end
end
Omega1=U.'*A*U;