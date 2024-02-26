
    img = imread('./XXX.png');

   
    [M,N]=size(img);

    temp=zeros(M,N);


    for m=1:M;
        for n=1:N;
            if img(m,n)==0;
               j=1;
            else
               j=img(m,n);
            end
            temp(j)=temp(j)+1;
        end
    end
    temp=temp./(M*N);

    
    EntropyResult=0;

    for j=1:length(temp)
        if temp(j)==0;
           EntropyResult=EntropyResult;
        else
           EntropyResult=EntropyResult-temp(j)*log2(temp(j));
        end
    end
    str = num2str(EntropyResult);
    disp(str);


