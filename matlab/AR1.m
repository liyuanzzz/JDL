p=400;
Omega1=[];
for i=1:p
    for j=1:p
        Omega1(i,j)=0.5^(abs(i-j));
    end
end
