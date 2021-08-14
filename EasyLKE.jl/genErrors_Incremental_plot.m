err1=load('errs1.csv');

figure;
plot(2:599,err1(1:end-1));
xlabel('No. Important Samples');
ylabel('||S_i-S_{svd}||^2');