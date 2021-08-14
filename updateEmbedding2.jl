
include("G:/Code/ILKE_experiments_julia/Arrowhead.jl/src/Arrowhead.jl")
# include("Arrowhead.jl/src/DoubleDouble.jl")
# include("Arrowhead.jl/src/arrowhead1.jl")
# include("Arrowhead.jl/src/arrowhead3.jl")
# include("Arrowhead.jl/src/arrowhead4.jl")
# include("Arrowhead.jl/src/arrowhead5.jl")
# include("Arrowhead.jl/src/arrowhead6.jl")
# include("Arrowhead.jl/src/arrowhead7.jl")
# include("Arrowhead.jl/src/arrowhead8.jl")
using .Arrowhead
#using Main.Arrowhead
# Implementing Hallgren Rank 1 Eig Update for samples Xnew
function updateEmbeddingLKE(Snew::Vector{Float64},Vnew::Array{Float64,2},ρₛ::Vector{Vector{Float64}})
    for nn in 1:length(ρₛ)
        #ρₙ= ρₛ[nn]
        Snew,Vnew= rankOneUpdateProc(Snew,Vnew,ρₛ[nn])#ρₙ)
    end
    return Snew,Vnew
end

function updateEmbeddingLKEah(Snew::Vector{Float64},Vnew::Array{Float64,2},ρₛ::Vector{Vector{Float64}})
    for nn in 1:length(ρₛ)
        #ρₙ= ρₛ[nn]
        Snew,Vnew= arrowheadUpdateProc(Snew,Vnew,ρₛ[nn])#ρₙ)
    end
    return Snew,Vnew
end

function arrowheadUpdateProc(S::Vector{Float64},V::Array{Float64,2},rho::Vector{Float64})
    L, U    = expandEigensystem(S, V, rho[end]/2)
    Brho    = U'*rho
    sym1    = Main.Arrowhead.SymArrow(L[1:end-1],Brho[1:end-1],rho[end],length(L))
    E,info  = eigen(sym1)
    #Vnew    = U*E.vectors

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

    #k1 = convert(Vector{Float64},k1)
    #k0 = convert(Vector{Float64},k0)

    Ln,Un = updateEigensystem(L, U, k1, sigma)
    Snew,Vnew = updateEigensystem(Ln, Un, k0, -sigma)

    return Snew,Vnew

end

function expandEigensystem(L::Vector{Float64},U::Array{Float64,2},e::Float64)

    m = length(L)
    L = cat(L, e, dims=1)
    U = cat(U, zeros(m,1),dims=2)
    U = cat(U,zeros(1,m+1),dims=1)
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
