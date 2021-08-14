include("EasyLKE.jl")

using Main.EasyLKE
using Random
using KernelFunctions
using LinearAlgebra
using Plots
using CSV, DelimitedFiles
# Specifying Kernel Function and Generating Embedding Map Components
kfnc = SqExponentialKernel()#
kfnc = transform(kfnc,2.0)
D=600
c=2000

Data=rand(D,c)
let ZR1=reshape(Data[:,1],D,1)
    W=kernelmatrix(kfnc,ZR1,ZR1,obsdim=2)
    V1,S1=svd(W)
    err1=zeros(c-1,1)
    for ii = 2:c
        ZRnew=reshape(Data[:,ii],D,1)
        W=kernelmatrix(kfnc,cat(ZR1,ZRnew,dims=2),cat(ZR1,ZRnew,dims=2),obsdim=2)
        Vs,Ss=svd(W)
        V1,S1,ZR1   = EasyLKE.incrementLKE(V1, S1, ZR1, ZRnew, kfnc)
        E= diagm(Ss)-diagm(S1)#broadcast(abs,S1)-broadcast(abs,Ss)
        err1[ii-1]=sqrt(tr(E'*E))
    end
    writedlm("G:/Code/EasyLKE.jl/errs_new.csv",err1,',')
    #plot(2:c-2,err1[1:end-2],xaxis = ("No. Important Samples, c", (1,599), 1:50:599, :log, :flip, font(12, "Courier")), yaxis = ("||Vₛ - Vᵢ||²", (0,0.1), 0:5e-3:0.1, :log, :flip, font(12, "Courier")))
    plot(2:c-2,err1[1:end-2])
    xlabel!("No. Important Samples, c")
    ylabel!("||Vₛ - Vᵢ||²")
end
