include("EasyLKE.jl")
include("G:/Code/EasySRC.jl/EasyData.jl")
include("G:/Code/EasySRC.jl/EasySRC.jl")

using CSV, DelimitedFiles
using Main.EasyLKE
using Main.EasyData
using Main.EasySRC
using Random
using KernelFunctions
using LinearAlgebra
## Parameters for RUnning Our desired Experiment
use_saved   = false#true#%false
base_path   = "G:/Code/EasyLKE.jl/Models/"
model_name  = "LKE-CIFAR-10-26-20.csv"

# Names of some model and embedding parameters that will be saved
V1S1Z1name  = string(base_path,"V1S1Z1-",model_name)
#V2S2Z2name  = string(base_path,"V2S2Z2-",model_name)
Model1name  = string(base_path,"M1-",model_name)
#Model2name  = string(base_path,"M2-",model_name)

## Importing Datasets (taking only a fraction of the total available)
dat1,labels1 = importData("CIFAR10",1.0)
#dat2,labels2 = importData("USPS",1.0)

#Normalize Columns of datasets
dat1 = dat1./(sum(dat1.^2,dims=1).^(1/2))
#dat2 = dat2./(sum(dat2.^2,dims=1).^(1/2))

## Splitting Datasets into Training and Testing
# Training percentages for dat1 and dat2
tp1 = 5/6#1/3#1/5
#tp2 = 0.1#1/5

Ztr1,ytr1,Zte1,yte1 = trainTestData(dat1,labels1,tp1)
#Ztr2,ytr2,Zte2,yte2 = trainTestData(dat2,labels2,tp2)
display("Data Separated into Training and Testing Sets...")

## Embedding Samples using a Random set of Samples for ZR
c = 4096#512      # c is the number of important samples (the column dimension of ZR)
# Selecting Important Samples
ZR1ind  = randperm(size(Ztr1,2))[1:c]
ZR1     = Ztr1[:,ZR1ind]

# Specifying Kernel Function and Generating Embedding Map Components
kfnc = SqExponentialKernel()#
kfnc = transform(kfnc,2.0)
# Some Other Kernel Choices:  LaplacianKernel() MaternKernel(), Matern32Kernel(), Matern52Kernel() LinearKernel(c=0.5)#PolynomialKernel(c=2.0,d=2.0)
# If we aren't using a saved embedding then resolve the embedding
if use_saved
    V1S1Z1  = readdlm(V1S1Z1name,',')
    c       = size(V1S1Z1,1)
    V1      = V1S1Z1[:,1:c]
    S1      = vec(V1S1Z1[:,c+1])
    ZR1     = V1S1Z1[:,c+2:end]'
else
    V1,S1,Ctr1  = embedLKE(Ztr1, ZR1, kfnc)
    V1S1Z1      = cat(V1,S1,ZR1',dims=2)
    writedlm(V1S1Z1name,  V1S1Z1, ',')
end
# Generating the C matrices for other training and testing sets
Ctr1=kernelmatrix(kfnc,Ztr1,ZR1,obsdim=2)
#Ctr2=kernelmatrix(kfnc,Ztr2,ZR1,obsdim=2)
Cte1=kernelmatrix(kfnc,Zte1,ZR1,obsdim=2)
#Cte2=kernelmatrix(kfnc,Zte2,ZR1,obsdim=2)
# Selecting the rank of the subspace that samples will be projected into
k =convert(Int64,floor(c/4))

Vk1=V1[:,1:k]
Sk1=S1[1:k]
# Generating the Vitrual Samples
Ftr1=diagm(Sk1)^(-1/2)*Vk1'*Ctr1'
Fte1=diagm(Sk1)^(-1/2)*Vk1'*Cte1'
#Ftr2=diagm(Sk1)^(-1/2)*Vk1'*Ctr2'
#Fte2=diagm(Sk1)^(-1/2)*Vk1'*Cte2'

Utr1=Ftr1
#Utr2=Ftr2
Ute1=Fte1
#Ute2=Fte2
## Parameters for Baseline Model
struct params
    K::Int64
    max_iter::Int64
    tau::Int64
    SA::Float64
    data_init::Bool
    des_avg_err::Float64
end

tau             = 30
max_iter        = 10#30
K               = 800#800
des_avg_err     = 1e-7
learning_params = params(K,max_iter,tau,tau/K,true,des_avg_err)
learning_method = "KSVD"

## Generate Baseline Model using MNIST training only
if !use_saved
    Model1=EasySRC.genSRCModel(learning_method,learning_params,Utr1,ytr1)
    # Saving Generated Model
    EasySRC.saveSRCModel(Model1name,Model1)
else
    Model1=EasySRC.genSRCModel(Model1name)
end
## Make Inferences on the MNIST test data
stats,decs=EasySRC.classify(Ute1,Model1)
display(sum(decs.==yte1)/length(yte1))
EasySRC.confusionSRC(yte1,decs)
