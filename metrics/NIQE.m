
path = 'XXX/XXX/'; 
files=dir(fullfile(path,'*.png'));
len=size(files,1)
myniqe = [];

for i=1:len
    filename=strcat(path,files(i).name);  
    img=imread(filename); 
    Qualty_Val = niqe(img);
    myniqe(i)=roundn(Qualty_Val,-4);
end

sum(myniqe)/len