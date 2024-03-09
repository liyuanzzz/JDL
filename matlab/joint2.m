p=200;
P1=[];
P2=[];
a=[];
a1=[];
a2=[];
Sigma1=[];
Sigma2=[];
for i=1:p
    A(i,i)=unifrnd(0.2,0.6);
    for j=1:p
        k=binornd(1,0.2);
        m=binornd(1,0.2);
        if k==1
            P1(i,j)=unifrnd(0,1);
        end
        if m==1
            P2(i,j)=unifrnd(0,1);
        end
    end
    for j=1:4
        a(i,j)=binornd(1,0.4);
    end
    for j=1:2
        a1(i,j)=binornd(1,0.4);
        a2(i,j)=binornd(1,0.4);
    end
end
Sigma1=0.5*eye(p)+0.2*a*a'+0.2*a1*a1'+0.01*(P1+P1');
Sigma2=0.5*eye(p)+0.2*a*a'+0.2*a2*a2'+0.01*(P2+P2');