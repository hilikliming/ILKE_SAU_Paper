module EasyLKE
export embedLKE, incrementLKE#, decrementLKE #To Do

#include("G:/Code/ILKE_experiments_julia/updateEmbedding2.jl")
include("../Arrowhead.jl/src/Arrowhead.jl")
using KernelFunctions
using .Arrowhead


function embedLKE(Z::AbstractMatrix, ZR::AbstractMatrix, kfnc::Kernel )
    W=kernelmatrix(kfnc,ZR,obsdim=2)
    C=kernelmatrix(kfnc,Z,ZR,obsdim=2)
    V,S=svd(W)
    return V, S, C
end

function incrementLKE(V_old::AbstractMatrix, S_old::Vector{Float64}, ZR_old::AbstractMatrix, Znew::AbstractMatrix, kfnc::Kernel)
    xnew  = Znew
    Xnewp = ZR_old#Array{Float64}(undef, size(xnew,1), 0)
    rhos  = Vector{Vector{Float64}}(undef,size(xnew,2))
    for nn in 1:size(xnew,2)
        Xnewp       = cat(Xnewp,xnew[:,nn],dims=2)
        rhos[nn]    = vec(kernelmatrix(kfnc,reshape(xnew[:,nn],size(xnew,1),1),Xnewp, obsdim=2)) #simpleKernelEval!(reshape(xnew[:,nn],length(xnew[:,nn]),1),Xnewp,kfnc!)
    end

    ZR1=ZR_old
    Snew,Vnew=updateEmbeddingLKEah(S_old,V_old,rhos)
    ZR2=cat(ZR1,xnew,dims=2)
    return Vnew,Snew,ZR2

end

#To-Do
#function decrementLKE(V_old::AbstractMatrix, S_old::AbstractMatrix, ZR_old::AbstractMatrix, drop_indices::Vector{Int64}, kfnc::Kernel)
#    ## Going one sample at a time, perform the Rank One Downdate procedure (extension of Hallgren method)
#end

## Methods Below are utilized in model updated of LKE embedding
# Implementing Hallgren Rank 1 Eig Update for samples Xnew
function updateEmbeddingLKE(Snew::Vector{Float64},Vnew::Array{Float64,2},ρₛ::Vector{Vector{Float64}})
    for nn in 1:length(ρₛ)
        Snew,Vnew= rankOneUpdateProc(Snew,Vnew,ρₛ[nn])
    end
    return Snew,Vnew
end

function updateEmbeddingLKEah(Snew::Vector{Float64},Vnew::Array{Float64,2},ρₛ::Vector{Vector{Float64}})
    for nn in 1:length(ρₛ)
        Snew,Vnew= arrowheadUpdateProc(Snew,Vnew,ρₛ[nn])
    end
    return Snew,Vnew
end

function arrowheadUpdateProc(S::Vector{Float64},V::Array{Float64,2},rho::Vector{Float64})
    L, U    = expandEigensystem(S, V, rho[end]/2)
    Brho    = U'*rho
    #sym1    = SymArrow(L[1:end-1],Brho[1:end-1],rho[end],length(L))
    #E,info  = eigen(sym1)
    E,info  = eigen(SymArrow(L[1:end-1],Brho[1:end-1],rho[end],length(L)))
    #E  = eigen(SymArrow(L[1:end-1],Brho[1:end-1],rho[end],length(L)))
    #display(typeof(E))
    #return E[1],U*E[2]
    return E.values, U*E.vectors

end

function rankOneUpdateProc(S::Vector{Float64},V::Array{Float64,2},k::Vector{Float64})
    sigma   = 4/k[end]
    k1      = deepcopy(k)
    k0      = deepcopy(k)

    k1[end] = k1[end]/2
    k0[end] = k0[end]/4

    L, U    = expandEigensystem(S, V, k0[end])
    idxs    = sortperm(L,rev=true)

    L   = sort(L,rev=true)
    U   = U[:,idxs]

    Ln,Un = updateEigensystem(L, U, k1, sigma)
    Snew,Vnew = updateEigensystem(Ln, Un, k0, -sigma)

    return Snew,Vnew

end

function expandEigensystem(L::Vector{Float64},U::Array{Float64,2},e::Float64)

    m = length(L)
    L = cat(L, e, dims=1)
    #U = cat(U, zeros(m,1),dims=2)
    #U = cat(U,zeros(1,m+1),dims=1)
    U = cat(cat(U, zeros(m,1),dims=2),zeros(1,m+1),dims=1)
    U[m+1,m+1] = 1
    return L, U
end

function  updateEigensystem(L::Vector{Float64},U::Array{Float64,2},v::Vector{Float64},sigma::Float64)
    z = U'*v

    sym1=SymDPR1(L,z,sigma)
    τ1=[1e3,10.0*length(sym1.D),1e3,1e3,1e3].*1e-1

    E,info=eigen(sym1,τ1)

    #Lnew    = E.values
    #Ut      = E.vectors
    #Unew    = U*Ut

    return E.values, U*E.vectors #Unew
end


end ## End of Module
