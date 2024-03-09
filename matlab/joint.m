p=400;
A=[];
a=[];
a1=[];
a2=[];
Sigma1=[];
Sigma2=[];
for i=1:p
    A(i,i)=unifrnd(0.2,0.6);
    for j=1:4
        a(i,j)=binornd(1,0.4);
    end
    for j=1:2
        a1(i,j)=binornd(1,0.4);
        a2(i,j)=binornd(1,0.4);
    end
end
Sigma1=A+0.2*a*a'+0.2*a1*a1';
Sigma2=A+0.2*a*a'+0.2*a2*a2';