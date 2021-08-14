module EasySRC
export genSRCModel, saveSRCModel, classify, confusionSRC, incIKSVD, incIDUO, SRCModel

import MLBase
include("../IKSVD.jl/src/IKSVD.jl")
include("../IDUO.jl/src/IDUO.jl")

using ProgressMeter, Base.Threads, Random, SparseArrays, LinearAlgebra
using CSV, DelimitedFiles, DataFrames, DSP, JLD
#using Main.IKSVD

struct SRCModel
    D::Array{Array{Float64,2},1}
    n_classes::Int64
    sparsity::Int64
    des_avg_err::Float64
end


function genSRCModel(learning_method::String, learning_params::Any, Data::AbstractMatrix, labels::Vector{Int64})
    if learning_method == "KSVD"
        params      = learning_params
        K           = params.K
        max_iter    = params.max_iter
        tau         = params.tau
        SA          = params.SA
        data_init   = params.data_init
        des_avg_err = params.des_avg_err
        M,N         = size(Data)
        u_labels    = unique(labels)
        u_labels    = sort(u_labels)
        #display(u_labels)

        n_labels    = length(u_labels)
        Ds          = Array{Float64,2}[]
        Model1      = SRCModel(Ds,n_labels,tau,des_avg_err)
        # Loop through all the classes and generte a dictionary using the specifed parameters
        for nn in 1:n_labels
            LiM = in.(labels, [u_labels[nn]])
            println(string("Training Dictionary No. ",string(nn),"\n"))
            @time D, X = IKSVD.ksvd(Data[:,LiM], K, max_iter=max_iter, max_iter_mp = tau, sparsity_allowance = SA,data_init=data_init,des_avg_err=des_avg_err)
            #Model1.D[:,:,nn]=D
            push!(Model1.D,D) #Add the next dictionary to the model.D vector (a vector of dictionary matrices)
        end
    else
        display("Invalid Learning Method Selected")
        Model1=[]
    end
    return Model1
end

# Overload function to allow generation from saved file
# function genSRCModel(filename::String)
# end
function genSRCModel(model_path::String)
    Dread   = load(model_path)#readdlm(model_path)
    n0      = Dread["n0"]#convert(Int64,Dread[1,1])
    s       = Dread["s"]#convert(Int64,Dread[2,1])
    err     = Dread["err"]#convert(Float64,Dread[4,1])
    D       = Dread["Ds"]#convert(Array{Float64,3},Dread[:,:,2:end])
    Model1  = SRCModel(D,n0,s,err)
    return Model1
end

function saveSRCModel(model_path::String, Model::SRCModel)
    if isfile(model_path)
        rm(model_path)
    end
    save(model_path,"n0", Model.n_classes,"s",Model.sparsity, "err", Model.des_avg_err, "Ds", Model.D)
end


function classify(Data::Array{Float64,2}, Model::SRCModel, fpd::Int64)
    nc=Model.n_classes
    stats=zeros(nc,size(Data,2))
    Xte=Data
    for nn=1:nc
        D = Model.D[nn]#reshape(Model.D[:,:,nn],size(Model.D,1),size(Model.D,2))
        XOMP=IKSVD.omp(Xte,D,Model.sparsity,Model.des_avg_err)
        stats[nn,:]= [(Xte[:,ii]-D*XOMP[:,ii])'*(Xte[:,ii]-D*XOMP[:,ii]) for ii in 1:size(Xte,2)]
    end

    bb=vec(ones(fpd,1))./fpd
    aa=vec([1])
    for row=1:size(stats,1)
       stats[row,:]=filt(bb,aa,vec(stats[row,:]))
    end

    decs=argmin(stats,dims=1)
    decs=[ i[1] for i in decs ] #or map(i->i[2], decs)
    decs=vec(decs)
    return stats,decs
end

function confusionSRC(y::Vector,yhat::Vector)
    uni_m=unique(y)
    C1=MLBase.confusmat(length(uni_m),y,yhat)
    C1=convert(Array{Int64},C1)
    ss=sum(C1,dims=2)
    C1=C1./ss
    display(C1)
    return C1
end

function incIKSVD(Model::SRCModel,Data::AbstractMatrix,labels::Vector{Int64}, learning_params::Any, K1::Int64)
    display(string("Incremental Updating via IKSVD using ",size(Data,2)," samples"))
    params      = learning_params

    max_iter    = params.max_iter
    tau         = params.tau
    SA          = params.SA
    data_init   = params.data_init
    des_avg_err = params.des_avg_err
    uni_m= sort(unique(labels))
    nti  = length(uni_m)#Model.n_classes
    #Knew = Model.K+K1

    d    = size(Model.D,1)
    Ds   = Array{Float64,2}[]#zeros(d,Knew,Model.n_classes)
    for nn=1:nti
        Dold=Model.D[nn]#reshape(Model.D[:,:,nn],d,Model.K)
        LiM = in.(labels, [uni_m[nn]])
        if !isempty(LiM)
            @time D, X = IKSVD.iksvd(Data[:,LiM], Dold, K1, max_iter=max_iter, max_iter_mp = tau, sparsity_allowance = SA, data_init=data_init, des_avg_err=des_avg_err)
            push!(Ds,D)
        else
            push!(Ds,Dold)
        end
    end

    Model2=SRCModel(Ds,Model.n_classes,tau,des_avg_err)
    return Model2
    #end
end

function incIDUO(Model::SRCModel,Data::AbstractMatrix,labels::Vector{Int64}, learning_params::Any, K1::Int64)
    display(string("Incremental Updating via IDUO using ",size(Data,2)," samples"))
    params      = learning_params

    max_iter    = params.max_iter
    tau         = params.tau
    SA          = params.SA
    data_init   = params.data_init
    des_avg_err = params.des_avg_err
    uni_m= sort(unique(labels))
    nti  = length(uni_m)#Model.n_classes


    #Knew = Model.K+K1

    d    = size(Model.D,1)
    Ds   = Array{Float64,2}[]#zeros(d,Knew,Model.n_classes)
    for nn=1:nti
        Dold=Model.D[nn]#reshape(Model.D[:,:,nn],d,Model.K)
        LiM = in.(labels, [uni_m[nn]])
        if !isempty(LiM)
            @time D, X = IDUO.iduo(Data[:,LiM], Dold, K1, max_iter=max_iter, max_iter_mp = tau, data_init=data_init, des_avg_err=des_avg_err)
            push!(Ds,D)#Ds[:,:,nn]=D
        else
            push!(Ds,Dold)
        end
    end

    Model2=SRCModel(Ds,Model.n_classes,tau,des_avg_err)
    return Model2
    #end
end


end ## End of Module
