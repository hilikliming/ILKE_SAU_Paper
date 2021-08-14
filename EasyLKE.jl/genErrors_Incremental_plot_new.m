err1=load('errs_new.csv');

figure;
plot(2:1999,err1(1:end-1));
xlabel('No. Important Samples');
ylabel('|| S_i  -  S_{svd} ||_F');