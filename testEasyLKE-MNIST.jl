include("EasyLKE.jl/EasyLKE.jl")
include("EasySRC.jl/EasyData.jl")
include("EasySRC.jl/EasySRC.jl")

using CSV, DelimitedFiles
using Main.EasyLKE
using Main.EasyData
using Main.EasySRC
using Random
using KernelFunctions
using LinearAlgebra


## Parameters for RUnning Our desired Experiment
use_saved1   = false#false#false#true#true#%false
use_saved2   = false#true#false#true#true

baseline_testing = false
t_alone_testing  = true
apd              = 1

# Change this to the directory where you would like to save model components
base_path   = "G:/Code/Fixing ILKE Repo/ILKE_paper/Models/"
model_name  = "LKE-MNIST-12-28-20.jld"#"LKE-UXOs-12-14-20.csv"#
mat_name    = "LKE-MNIST-12-28-20.csv"

# Names of some model and embedding parameters that will be saved
V1S1Z1name  = string(base_path,"V1S1Z1-",mat_name)
V2S2Z2name  = string(base_path,"V2S2Z2-",mat_name)
V3S3Z3name  = string(base_path,"V3S3Z3-",mat_name)

Model1name  = string(base_path,"M1-",model_name)
Model2name  = string(base_path,"M2-IK-TREX-",model_name)
Model3name  = string(base_path,"M3-IK-BAYEX-",model_name)
## Importing Datasets (taking only a fraction of the total available)

# Demo with small portion for quicker training.
dat1,labels1 = importData("MNIST",0.1)#importData("MNIST",0.8)#
dat2,labels2 = importData("USPS",0.5)#importData("USPS",1.0)#
#dat3,labels3 = importData("BAYEX14",1.0)#

#Normalize Columns of datasets
dat1 = dat1./(sum(dat1.^2,dims=1).^(1/2))
dat2 = dat2./(sum(dat2.^2,dims=1).^(1/2))
#dat3 = dat3./(sum(dat3.^2,dims=1).^(1/2))


## Splitting Datasets into Training and Testing
# Training percentages for dat1 and dat2
tp1 = 6/7
tp2 = 0.05
#tp3 = 0.05

# Split the samples into a training and testing portion
Ztr1,ytr1,Zte1,yte1 = trainTestData(dat1,labels1,tp1)
Ztr2,ytr2,Zte2,yte2 = trainTestData(dat2,labels2,tp2)
#Ztr3,ytr3,Zte3,yte3 = trainTestData(dat3,labels3,tp3)

display("Data Separated into Training and Testing Sets...")

## Embedding Samples using a Random set of Samples for ZR
c = 1200#1800 #1200     # c is the number of important samples (the column dimension of ZR)

# Selecting Important Samples from dataset 1 (MNIST)
ZR1ind  = randperm(size(Ztr1,2))[1:c]
ZR1     = Ztr1[:,ZR1ind]

# Specifying Kernel Function and Generating Embedding Map Components
kfnc = 2.0*SqExponentialKernel()#
# ^^transform is depricated, scale with nottion above^^
#kfnc = transform(kfnc,4.0)
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
#Ctr3=kernelmatrix(kfnc,Ztr3,ZR1,obsdim=2)

Cte1=kernelmatrix(kfnc,Zte1,ZR1,obsdim=2)
Cte2=kernelmatrix(kfnc,Zte2,ZR1,obsdim=2)
#Cte3=kernelmatrix(kfnc,Zte3,ZR1,obsdim=2)

# Selecting the rank of the subspace that samples will be projected into
k = 300#400

Vk1=V1[:,1:k]
Sk1=S1[1:k]
# Generating the Vitrual Samples
Ftr1=diagm(Sk1)^(-1/2)*Vk1'*Ctr1'
Fte1=diagm(Sk1)^(-1/2)*Vk1'*Cte1'

Ftr2=diagm(Sk1)^(-1/2)*Vk1'*Ctr2'
Fte2=diagm(Sk1)^(-1/2)*Vk1'*Cte2'

#Ftr3=diagm(Sk1)^(-1/2)*Vk1'*Ctr3'
#Fte3=diagm(Sk1)^(-1/2)*Vk1'*Cte3'

Utr1=Ftr1
Utr2=Ftr2
#Utr3=Ftr3

Ute1=Fte1
Ute2=Fte2
#Ute3=Fte3

## Parameters for Baseline Model
struct params
    K::Int64
    max_iter::Int64
    tau::Int64
    SA::Float64
    data_init::Bool
    des_avg_err::Float64
end

# Some parameters to train a model quickly, or you could use the parameters in the paper.
tau             = 15          # The number of non-zero coefficients in OMP estima
max_iter        = 10#20#30tes # THe number of training iterations for K-SVD learning
K               = 200         # THe number of K-SVD atoms learned per dictionary (L in our paper)
des_avg_err     = 1e-6        # The desired average error in reconstruction
learning_params = params(K,max_iter,tau,tau/K,true,des_avg_err) # structify those params
learning_method = "KSVD"      # Specify the learning method (KSVD is supported right now but more to come!)

## Generate Baseline Model using MNIST training only
if !use_saved1
    Model1=EasySRC.genSRCModel(learning_method,learning_params,Utr1,ytr1)
    # Saving Generated Model
    EasySRC.saveSRCModel(Model1name,Model1)
else
    Model1=EasySRC.genSRCModel(Model1name)
end


## Make Inferences on the MNIST test data

if baseline_testing
    println("MNIST Performance, Model 1")
    #println("MNIST Performance, Model 1")
    stats,decs=EasySRC.classify(Ute1,Model1,apd)
    display(sum(decs.==yte1)/length(yte1))
    C1=EasySRC.confusionSRC(yte1,decs)
    display(diag(C1)')

    println("USPS Performance, Model 1")
    #println("USPS Performance, Model 1")
    # Make Inferences on the TREX test data
    stats,decs=EasySRC.classify(Ute2,Model1,apd)
    display(sum(decs.==yte2)/length(yte2))
    C2=EasySRC.confusionSRC(yte2,decs)
    display(diag(C2)')

    # println("BAYEX Performance, Model 1")
    # # Make Inferences on the BAYEX test data
    # stats,decs=EasySRC.classify(Ute3,Model1,apd)
    # display(sum(decs.==yte3)/length(yte3))
    # C3=EasySRC.confusionSRC(yte3,decs)
    # display(diag(C3)')
else
    display("Baseline Testing Skipped!")
end
## Update Embedding using limited amount of USPS
b       = 205#size(Ztr2,2)#345
ZR2ind  = randperm(size(Ztr2,2))[1:b]
ZRnew   = Ztr2[:,ZR2ind]

if use_saved2
    V2S2Z2  = readdlm(V2S2Z2name,',')
    c       = size(V2S2Z2,1)
    V2      = V2S2Z2[:,1:c]
    S2      = vec(V2S2Z2[:,c+1])
    ZR2     = V2S2Z2[:,c+2:end]'
    k₂      = k#+20##k+60c
else
    V2,S2,ZR2   = EasyLKE.incrementLKE(V1, S1, ZR1, ZRnew, kfnc)
    display("Embedding Update Complete!")
    # Saving Updated embedding components
    V2S2Z2      = cat(V2,S2,ZR2',dims=2)
    k₂  = k#+20##k+60
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
#Ctr3    = kernelmatrix(kfnc,Ztr3,ZR2,obsdim=2)

Cte1    = kernelmatrix(kfnc,Zte1,ZR2,obsdim=2)
Cte2    = kernelmatrix(kfnc,Zte2,ZR2,obsdim=2)
#Cte3    = kernelmatrix(kfnc,Zte3,ZR2,obsdim=2)

# Generating the Corresponding F matrices for updated embedding
Utr1    = diagm(Sk2)^(-1/2)*Vk2'*Ctr1'
Utr2    = diagm(Sk2)^(-1/2)*Vk2'*Ctr2'
#Utr3    = diagm(Sk2)^(-1/2)*Vk2'*Ctr3'

Ute1    = diagm(Sk2)^(-1/2)*Vk2'*Cte1'
Ute2    = diagm(Sk2)^(-1/2)*Vk2'*Cte2'
#Ute3    = diagm(Sk2)^(-1/2)*Vk2'*Cte3'

# Transform Old Model Dictionaries to new space
Dold    = Model1.D
n_dicts = length(Dold)
Dnew    =  Array{Float64,2}[]#zeros(k₂,K,n_dicts)

for class in 1:n_dicts
    DD              = T*Dold[class]
    push!(Dnew,DD)
end
# Update Model dictionaries to new embedding
Model_T=EasySRC.SRCModel(Dnew,Model1.n_classes,Model1.sparsity, Model1.des_avg_err)
display("Dictionaries Have Been Transformed to New Embedding")
## Update Model using limited amount of USPS
K1    =   200#30#10     # Number of additional atoms to add.

# Train Incremental update to dictionaries using IDUO or IKSVD
if use_saved2
    Model2  = EasySRC.genSRCModel(Model2name)
else
    Model2  = EasySRC.incIKSVD(Model_T,Utr2,ytr2,learning_params,K1)
    #Model2  = EasySRC.incIDUO(Model_T,Utr2,ytr2,learning_params,K1)
    EasySRC.saveSRCModel(Model2name,Model2)
end

if t_alone_testing
    # Classifying Dataset 1 using T-alone model
    stats,decs  =   EasySRC.classify(Ute1,Model_T,apd)
    display("T ALONE Dataset 1: MNIST")
    display(sum(decs.==yte1)/length(yte1))
    C3=EasySRC.confusionSRC(yte1,decs)
    display(diag(C3)')
    # Classifying Dataset 2 using T-alone model
    stats,decs  =   EasySRC.classify(Ute2,Model_T,apd)
    display("T ALONE Dataset 2: USPS")
    display(sum(decs.==yte2)/length(yte2))
    C4=EasySRC.confusionSRC(yte2,decs)
    display(diag(C4)')
    # # Classifying Dataset 3 using T-alone model
    # stats,decs  =   EasySRC.classify(Ute3,Model_T,apd)
    # display("T ALONE Dataset 3: BAYEX14")
    # display(sum(decs.==yte3)/length(yte3))
    # C5=EasySRC.confusionSRC(yte3,decs)
    # display(diag(C5)')

else
    display("T-alone testing skipped!")
end

# Classifying Dataset 1 using T+IDUO
stats,decs  =   EasySRC.classify(Ute1,Model2,apd)
display("T + IDUO Dataset 1: MNIST")
display(sum(decs.==yte1)/length(yte1))
C6=EasySRC.confusionSRC(yte1,decs)
display(diag(C6)')
# Classifying Dataset 2 using T+IDUO model
stats,decs  =   EasySRC.classify(Ute2,Model2,apd)
display("T + IDUO Dataset 2: USPS")
display(sum(decs.==yte2)/length(yte2))
C7=EasySRC.confusionSRC(yte2,decs)
display(diag(C7)')
