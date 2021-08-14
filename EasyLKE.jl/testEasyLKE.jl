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
use_saved1   = true#true#true#%false
use_saved2   = true#true#true
use_saved3   = true#true
base_path   = "G:/Code/EasyLKE.jl/Models/"
model_name  = "LKE-MNIST-11-19-20.csv"#"LKE-MNIST-11-16-20.csv"

# Names of some model and embedding parameters that will be saved
V1S1Z1name  = string(base_path,"V1S1Z1-",model_name)
V2S2Z2name  = string(base_path,"V2S2Z2-",model_name)
Model1name  = string(base_path,"M1-",model_name)
Model2name  = string(base_path,"M2-IK",model_name)
Model3name  = string(base_path,"M3-Retrain-",model_name)
## Importing Datasets (taking only a fraction of the total available)
dat1,labels1 = importData("MNIST",1.0)
dat2,labels2 = importData("USPS",1.0)

#Normalize Columns of datasets
dat1 = dat1./(sum(dat1.^2,dims=1).^(1/2))
dat2 = dat2./(sum(dat2.^2,dims=1).^(1/2))

## Splitting Datasets into Training and Testing
# Training percentages for dat1 and dat2
tp1 = 6/7#1/5
tp2 = 0.025#1/5

Ztr1,ytr1,Zte1,yte1 = trainTestData(dat1,labels1,tp1)
Ztr2,ytr2,Zte2,yte2 = trainTestData(dat2,labels2,tp2)
display("Data Separated into Training and Testing Sets...")

## Embedding Samples using a Random set of Samples for ZR
c = 1200      # c is the number of important samples (the column dimension of ZR)

# Selecting Important Samples
ZR1ind  = randperm(size(Ztr1,2))[1:c]
ZR1     = Ztr1[:,ZR1ind]

# Specifying Kernel Function and Generating Embedding Map Components
kfnc = SqExponentialKernel()#
kfnc = transform(kfnc,2.0)
# Some Other Kernel Choices:  LaplacianKernel() MaternKernel(), Matern32Kernel(), Matern52Kernel() LinearKernel(c=0.5)#PolynomialKernel(c=2.0,d=2.0)

# If we aren't using a saved embedding then re-solve the embedding
if use_saved1
    V1S1Z1  = readdlm(V1S1Z1name,',')
    c       = size(V1S1Z1,1)
    V1      = V1S1Z1[:,1:c]
    S1      = vec(V1S1Z1[:,c+1])
    ZR1     = V1S1Z1[:,c+2:end]'
else
    V1,S1,Ctr1  = EasyLKE.embedLKE(Ztr1, ZR1, kfnc)
    V1S1Z1      = cat(V1,S1,ZR1',dims=2)
    writedlm(V1S1Z1name,  V1S1Z1, ',')
end

# Generating the C matrices for other training and testing sets
Ctr1=kernelmatrix(kfnc,Ztr1,ZR1,obsdim=2)
Ctr2=kernelmatrix(kfnc,Ztr2,ZR1,obsdim=2)
Cte1=kernelmatrix(kfnc,Zte1,ZR1,obsdim=2)
Cte2=kernelmatrix(kfnc,Zte2,ZR1,obsdim=2)
# Selecting the rank of the subspace that samples will be projected into
k =400

Vk1=V1[:,1:k]
Sk1=S1[1:k]
# Generating the Vitrual Samples
Ftr1=diagm(Sk1)^(-1/2)*Vk1'*Ctr1'
Fte1=diagm(Sk1)^(-1/2)*Vk1'*Cte1'
Ftr2=diagm(Sk1)^(-1/2)*Vk1'*Ctr2'
Fte2=diagm(Sk1)^(-1/2)*Vk1'*Cte2'

Utr1=Ftr1
Utr2=Ftr2
Ute1=Fte1
Ute2=Fte2
## Parameters for Baseline Model
struct params
    K::Int64
    max_iter::Int64
    tau::Int64
    SA::Float64
    data_init::Bool
    des_avg_err::Float64
end

tau             = 10
max_iter        = 30
K               = 800
des_avg_err     = 1e-7
learning_params = params(K,max_iter,tau,tau/K,true,des_avg_err)
learning_method = "KSVD"

## Generate Baseline Model using MNIST training only
if !use_saved1
    Model1=EasySRC.genSRCModel(learning_method,learning_params,Utr1,ytr1)
    # Saving Generated Model
    EasySRC.saveSRCModel(Model1name,Model1)
else
    Model1=EasySRC.genSRCModel(Model1name)
end
## Make Inferences on the MNIST test data
stats,decs=EasySRC.classify(Ute1,Model1)
display(sum(decs.==yte1)/length(yte1))
C1=EasySRC.confusionSRC(yte1,decs)
display(diag(C1)')

# Make Inferences on the USPS test data
stats,decs=EasySRC.classify(Ute2,Model1)
display(sum(decs.==yte2)/length(yte2))
C2=EasySRC.confusionSRC(yte2,decs)
display(diag(C2)')
## Update Embedding using limited amount of USPS
b       = size(Ztr2,2)#345
ZR2ind  = randperm(size(Ztr2,2))[1:b]
ZRnew   = Ztr2[:,ZR2ind]

if use_saved2
    V2S2Z2  = readdlm(V2S2Z2name,',')
    c       = size(V2S2Z2,1)
    V2      = V2S2Z2[:,1:c]
    S2      = vec(V2S2Z2[:,c+1])
    ZR2     = V2S2Z2[:,c+2:end]'
    k₂      = k+60#c
else
    V2,S2,ZR2   = EasyLKE.incrementLKE(V1, S1, ZR1, ZRnew, kfnc)
    display("Embedding Update Complete!")
    # Saving Updated embedding components
    V2S2Z2      = cat(V2,S2,ZR2',dims=2)
    k₂  = k+60#k+b
    writedlm(V2S2Z2name,  V2S2Z2, ',')
end

# Generate New Kernel Approximation Wtilde (Wt) and augmented kernel matrix W₊
Wt  = V2*diagm(S2)*V2'
W₊  = kernelmatrix(kfnc,ZR1,ZR2,obsdim=2)

# Select New Subspace rank and the corresponding eigenvalues/vectors of Wt

Vk2 = V2[:,1:k₂]
Sk2 = S2[1:k₂]

# Generate Dictionary Transforming Matrix to update dictionaries to new embedding
T = diagm(Sk2)^(-1/2)*Vk2'*Wt*W₊'*(W₊*W₊')^(-1)*Vk1*diagm(Sk1)^(1/2)
# Generating the Updated C matrices using the new ZR (important samples) matrix.
Ctr1    = kernelmatrix(kfnc,Ztr1,ZR2,obsdim=2)
Ctr2    = kernelmatrix(kfnc,Ztr2,ZR2,obsdim=2)
Cte1    = kernelmatrix(kfnc,Zte1,ZR2,obsdim=2)
Cte2    = kernelmatrix(kfnc,Zte2,ZR2,obsdim=2)

# Generating the Corresponding F matrices for updated embedding
Utr1    = diagm(Sk2)^(-1/2)*Vk2'*Ctr1'
Utr2    = diagm(Sk2)^(-1/2)*Vk2'*Ctr2'
Ute1    = diagm(Sk2)^(-1/2)*Vk2'*Cte1'
Ute2    = diagm(Sk2)^(-1/2)*Vk2'*Cte2'

# Transform Old Model Dictionaries to new space
Dold    = Model1.D
n_dicts = size(Dold,3)
Dnew    = zeros(k₂,K,n_dicts)

for class in 1:n_dicts
    DD              = T*Dold[:,:,class]
    Dnew[:,:,class] = DD
end
# Update Model dictionaries to new embedding
Model_T=EasySRC.SRCModel(Model1.n_classes,Model1.sparsity, Model1.K, Model1.des_avg_err,Dnew)
display("Dictionaries Have Been Transformed to New Embedding")
## Update Model using limited amount of USPS
K1    =   10     # Number of additional atoms to add.

# Train Incremental update to dictionaries using IDUO or IKSVD
if use_saved2
    Model2  = EasySRC.genSRCModel(Model2name)
else
    #Model2  = EasySRC.incIKSVD(Model1,Utr2,ytr2,learning_params,K1)
    Model2  = EasySRC.incIDUO(Model_T,Utr2,ytr2,learning_params,K1)
    EasySRC.saveSRCModel(Model2name,Model2)
end

# Optinally Fully Re-train K-SVD dictionary using updated embedding features
if use_saved3
    Model3  = EasySRC.genSRCModel(Model3name)
else
    display("Re-Training Full K-SVD dictionaries in New embedding!!!!")
    Model3=EasySRC.genSRCModel(learning_method,learning_params,cat(Utr1,Utr2,dims=2),vec(cat(ytr1,ytr2,dims=1)))
    EasySRC.saveSRCModel(Model3name,Model3)
end



# Classifying Dataset 1 using T-alone model
stats,decs  =   EasySRC.classify(Ute1,Model_T)
display("T ALONE Dataset 1: MNIST")
display(sum(decs.==yte1)/length(yte1))
C3=EasySRC.confusionSRC(yte1,decs)
display(diag(C3)')
# Classifying Dataset 2 using T-alone model
stats,decs  =   EasySRC.classify(Ute2,Model_T)
display("T ALONE Dataset 1: USPS")
display(sum(decs.==yte2)/length(yte2))
C4=EasySRC.confusionSRC(yte2,decs)
display(diag(C4)')
# Classifying Dataset 1 using New model
stats,decs  =   EasySRC.classify(Ute1,Model2)
display("T + IDUO Dataset 1: MNIST")
display(sum(decs.==yte1)/length(yte1))
C5=EasySRC.confusionSRC(yte1,decs)
display(diag(C5)')
# Classifying Dataset 2 using New model
stats,decs  =   EasySRC.classify(Ute2,Model2)
display("T + IDUO Dataset 1: USPS")
display(sum(decs.==yte2)/length(yte2))
C6=EasySRC.confusionSRC(yte2,decs)
display(diag(C6)')

# Classifying Dataset 1 using Re-trained model
stats,decs  =   EasySRC.classify(Ute1,Model3)
display("K-SVD Re-train Dataset 1: MNIST")
display(sum(decs.==yte1)/length(yte1))
C7=EasySRC.confusionSRC(yte1,decs)
display(diag(C7)')

# Classifying Dataset 2 using Re-Trained model
stats,decs  =   EasySRC.classify(Ute2,Model3)
display("K-SVD Re-train Dataset 1: USPS")
display(sum(decs.==yte2)/length(yte2))
C8=EasySRC.confusionSRC(yte2,decs)
display(diag(C8)')
