include("EasyData.jl")
include("EasySRC.jl")

using Main.EasyData
using Main.EasySRC
using Random


## Parameters for RUnning Our desired Experiment
use_saved=true#false#true#%false

## Importing Datasets
dat1,labels1= importData("MNIST",1.0)
dat2,labels2= importData("USPS",1.0)

#Column Normalize Datasets
dat1=dat1./(sum(dat1.^2,dims=1).^(1/2))
dat2=dat2./(sum(dat2.^2,dims=1).^(1/2))

## Splitting Datasets into Training and Testing
# Training percentages for dat1 and dat2
tp1= 6/7#2/3
tp2= 1/5

Ztr1,ytr1,Zte1,yte1 = trainTestData(dat1,labels1,tp1)
Ztr2,ytr2,Zte2,yte2 = trainTestData(dat2,labels2,tp2)
display("Data Separated into Training and Testing Sets...")

## Parameters for Baseline Model
struct params
    K::Int64
    max_iter::Int64
    tau::Int64
    SA::Float64
    data_init::Bool
    des_avg_err::Float64
end
tau             = 5
max_iter        = 15
K               = 500#800
des_avg_err     = 1e-7

learning_params = params(K,max_iter,tau,tau/K,true,des_avg_err)
learning_method = "KSVD"

## Generate Baseline Model using MNIST training only
base_path="G:/Code/EasySRC.jl/"
model_name="MNIST-baseline-10-12-20.csv"
if !use_saved
    Model1=genSRCModel(learning_method,learning_params,Ztr1,ytr1)
    # Saving Generated Model
    saveSRCModel(string(base_path,model_name),Model1)
else
    Model1=genSRCModel(string(base_path,model_name))
end


## Make Inferences on the MNIST test data
stats,decs=classify(Zte1,Model1)
display(sum(decs.==yte1)/length(yte1))
confusionSRC(yte1,decs)

stats,decs=classify(Zte2,Model1)
display(sum(decs.==yte2)/length(yte2))
confusionSRC(yte2,decs)

## Update Model using limited amount of USPS

# Number of additional atoms to add.
K1=2
#Model3=incIKSVD(Model1,Ztr2,ytr2,learning_params,K1)
Model3=incIDUO(Model1,Ztr2,ytr2,learning_params,K1)
stats,decs=classify(Zte1,Model3)
display(sum(decs.==yte1)/length(yte1))
confusionSRC(yte1,decs)

stats,decs=classify(Zte2,Model3)
display(sum(decs.==yte2)/length(yte2))
confusionSRC(yte2,decs)
