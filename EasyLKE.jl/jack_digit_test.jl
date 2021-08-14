
include("EasyLKE.jl")
include("G:/Code/EasySRC.jl/EasyData.jl")
include("G:/Code/EasySRC.jl/EasySRC.jl")

using CSV, DelimitedFiles
using Main.EasyLKE
using Main.EasyData
using Main.EasySRC
using Random
using KernelFunctions
## Testing the lastest classifier with my handwritten digit
base_path   = "G:/Code/EasyLKE.jl/Models/"
model_name  = "LKE-MNIST-11-19-20.csv"#"LKE-MNIST-11-16-20.csv"

# Names of some model and embedding parameters that will be saved
V1S1Z1name  = string(base_path,"V1S1Z1-",model_name)
V2S2Z2name  = string(base_path,"V2S2Z2-",model_name)
Model1name  = string(base_path,"M1-",model_name)
Model2name  = string(base_path,"M2-",model_name)
Model3name  = string(base_path,"M3-Retrain-",model_name)

V2S2Z2  = readdlm(V2S2Z2name,',')
c       = size(V2S2Z2,1)
V2      = V2S2Z2[:,1:c]
S2      = vec(V2S2Z2[:,c+1])
ZR2     = V2S2Z2[:,c+2:end]'
k₂      = k+60#c

Vk2 = V2[:,1:k₂]
Sk2 = S2[1:k₂]

Model2=EasySRC.genSRCModel(Model2name)

## Silly test with an MSpaint digit lul...
if test_jack_img==true
    ji  = load("G:/Code/EasyLKE.jl/jack_digit.png")
    jd  = Gray.(ji)
    display(jd)
    jd  = convert(Array{Float64,2},jd)
    jd  = reshape(jd',size(jd,1)*size(jd,2),1)
    jd  = jd./(sum(jd.^2,dims=1)).^(1/2)
    cjd = kernelmatrix(kfnc,jd,ZR2,obsdim=2)
    fjd = diagm(Sk2)^(-1/2)*Vk2'*cjd'
    stat_jd,dec_jd  =   EasySRC.classify(fjd,Model2)
    display(dec_jd)
end
