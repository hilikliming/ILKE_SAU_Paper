module IKSVD
# This is an implementation of the Incremental K-SVD algorithm and is adapted
# from Ishita Takeshi's K-SVD implementation for Julia
# The original paper:
# IK-SVD: Dictionary Learning for Spatial Big Data via Incremental Atom Update
#
export iksvd, ksvd, matching_pursuit, omp
using ProgressMeter
using Base.Threads, Random, SparseArrays, LinearAlgebra


include("matching_pursuit.jl")

const default_sparsity_allowance = 0.9
const default_max_iter = 200
const default_max_iter_mp = 200

Random.seed!(1234)  # for stability of tests


function error_matrix(Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix, k::Int)
    # indices = [i for i in 1:size(D, 2) if i != k]
    indices = deleteat!(collect(1:size(D, 2)), k)
    return Y - D[:, indices] * X[indices, :]
end


function init_dictionary(n::Int, K::Int)
    # D must be a full-rank matrix
    D = rand(n, K)
    while rank(D) != min(n, K)
        D = rand(n, K)
    end

    @inbounds for k in 1:K
        D[:, k] ./= norm(@view(D[:, k]))
    end
    return D
end
## JJH 4-24-20 BEGIN
function init_dictionary_data(Data::AbstractMatrix, K::Int)
    # D must be a full-rank matrix
    n=size(Data,1)
    ind1=randperm(size(Data,2))[1:K]
    D = Data[:,ind1]
    while rank(D) != min(n, K)
        ind1=randperm(size(Data,2))[1:K]
        D = Data[:,ind1]
    end

    @inbounds for k in 1:K
        D[:, k] ./= norm(@view(D[:, k]))
        D[:, k] .= D[:,k].*sign(D[1,k]) # multiply in the sign of the first element. JJH
    end
    return D
end

# function find_better_atom(Data::AbstractMatrix,D::AbstractMatrix,k::Int,X::AbstractMatrix)
#     rel_inds=findall(!iszero,X[k,:])
#     if length(rel_inds)<1
#         E_mat=Data-D*X
#         Evec=sum(E_Mat.^2,dims=1)
#         ii=maxperm(Evec)
#         newAtom=Data[:,ii]
#         newAtom=normalize(newAtom)
#         newAtom=newAtom.*sign(newAtom[1])
#         X[k,:].=0
#         display("New Vector Added")
#     end
#
# end
## JJH 4-24-20 END

# Atom updates portion of K-SVD
function ksvd(Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix)
    N = size(Y, 2)
    kis=randperm(size(X,1))[1:end]
    for k in kis #1:size(X, 1)
        xₖ = X[k, :]
        # ignore if the k-th row is zeros
        all(iszero, xₖ) && continue

        # wₖ is the column indices where the k-th row of xₖ is non-zero,
        # which is equivalent to [i for i in N if xₖ[i] != 0]
        wₖ = findall(!iszero, xₖ)

        # Eₖ * Ωₖ implies a selection of error columns that
        # correspond to examples that use the atom D[:, k]
        Eₖ = error_matrix(Y, D, X, k)
        Ωₖ = sparse(wₖ, 1:length(wₖ), ones(length(wₖ)), N, length(wₖ))
        # Note that S is a vector that contains diagonal elements of
        # a matrix Δ such that Eₖ * Ωₖ == U * Δ * V.
        # Non-zero entries of X are set to
        # the first column of V multiplied by Δ(1, 1)
        U, S, V = svd(Eₖ * Ωₖ, full=true)
        D[:, k] = U[:, 1]
        X[k, wₖ] = V[:, 1] * S[1]
    end
    return D, X
end

function iksvd(Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix, m_atoms::Int)
    N  = size(Y, 2)
    kis= randperm(m_atoms)[1:end] .+ (size(X,1) - m_atoms)
    for k in kis #1:size(X, 1)
        xₖ = X[k, :]
        # ignore if the k-th row is zeros
        all(iszero, xₖ) && continue

        # wₖ is the column indices where the k-th row of xₖ is non-zero,
        # which is equivalent to [i for i in N if xₖ[i] != 0]
        wₖ = findall(!iszero, xₖ)

        # Eₖ * Ωₖ implies a selection of error columns that
        # correspond to examples that use the atom D[:, k]
        Eₖ = error_matrix(Y, D, X, k)
        Ωₖ = sparse(wₖ, 1:length(wₖ), ones(length(wₖ)), N, length(wₖ))
        # Note that S is a vector that contains diagonal elements of
        # a matrix Δ such that Eₖ * Ωₖ == U * Δ * V.
        # Non-zero entries of X are set to
        # the first column of V multiplied by Δ(1, 1)
        U, S, V = svd(Eₖ * Ωₖ, full=true)
        D[:, k] = U[:, 1]
        X[k, wₖ] = V[:, 1] * S[1]
    end
    return D, X
end

function iksvd(Y::AbstractMatrix, D_old::AbstractMatrix, m_atoms::Int;
              sparsity_allowance = default_sparsity_allowance,
              max_iter::Int = default_max_iter,
              max_iter_mp::Int = default_max_iter_mp, data_init::Bool, des_avg_err::Float64)


    n, N      = size(Y)
    m_atoms   = min(m_atoms,N)
    K1        = m_atoms
    ## Assuming D_old is the dictionary from Step 1, Step 2 of IKSVD
    ## requires us to select m_atoms number of samples to intialize the atoms
    ## d_1 ... d_m we do this using equation 16 from the paper

    # First we represent Y with the old dictionary and find those with worst
    # representation
    X_sparse    = omp(Y, D_old, max_iter_mp, des_avg_err)
    EY   = Y-D_old*X_sparse
    errs = diag(EY'EY) # Diagonal of gram is the vector of individual samples SqErr in representaiton

    s_errs=sort(errs,rev=true)     # sorted errors of the candidates
    i_errs=sortperm(errs,rev=true) # Indices of the highest error candidates

    i_cand=i_errs[1:m_atoms]

    ## Next we compute the entropy of the poorly represented samples
    H   =computeEntropy(X_sparse[:,i_cand])
    i_H =sortperm(H,rev=true)
    ## select m_atoms samples with highest entropy (16) should be an argmax in the paper...
    Dnew = Y[:,i_cand[i_H[1:m_atoms]]]
    D    = cat(D_old,Dnew, dims=2)

    # Initialize Progress meter
    p = Progress(max_iter)
    coeffCutoff=0.01
    maxIP=0.995
    minFracObs=1e-4
    X=0*X_sparse
    display(string("Beginning IK-SVD Training using ",size(Y,2), " Samples"))

    for i in 1:max_iter
        X_sparse    = omp(Y, D, max_iter_mp, des_avg_err)

        D, X        = iksvd(Y, D, X_sparse,m_atoms)#omp(Y, D, max_iter_mp, des_avg_err)) #X_sparse)
        avg_err     = norm((Y-D*X)'*(Y-D*X))/size(Y,2)
        ProgressMeter.next!(p; showvalues = [(:iter,i), (:avg_err,avg_err)])

        if avg_err <= des_avg_err
            display("Training Completed Early, Returning D and X")
            println(string("Final iteration: ",i))
            println(string("Average error: ",avg_err))
            return D, X
        end

        #display("Conditioning Dictionary (3)")

    end
    return D, X

end

# function computeEntropy(X::AbstractMatrix)
#     n,N = size(X)
#     p   = broadcast(abs,X)
#     for ii in 1:N
#         p[:,ii]= p[:,ii]/sum(p[:,ii])
#     end
#     lp = broadcast(log,p)
#     H = -sum(p.*lp,dims=1)
#     return vec(H)
# end
function computeEntropy(X::AbstractMatrix)
    n,N = size(X)
    p   = broadcast(abs,X)
    sp  = sum(p,dims=1)

    #ps=zeros(N,1)
    H=zeros(N,1)
    for ii in 1:N
        pC=p[:,ii]/sp[ii]
        idx_nz= .!in.(pC,[0])
        if isempty(idx_nz)
            H[ii]=0
        else
            H[ii]=-sum(pC[idx_nz].*broadcast(log,pC[idx_nz]))
        end
        # pt= p[:,ii]/sum(p[:,ii])
        # p[:,ii]=pt
    end
    # display(p)
    # lp = broadcast(log,p)
    # display(lp)
    # H = -sum(p.*lp,dims=1)
    if any(isnan,H)
        display("NaN detected in Entropy Computation")
        #pt=ones(size(pt))*1/(length(pt)^4)
    end
    return vec(H)
end

"""
    ksvd(Y::AbstractMatrix, n_atoms::Int;
         sparsity_allowance::Float64 = $default_sparsity_allowance,
         max_iter::Int = $default_max_iter,
         max_iter_mp::Int = $default_max_iter_mp)

Run K-SVD that designs an efficient dictionary D for sparse representations,
and returns X such that DX = Y or DX ≈ Y.

```
# Arguments
* `sparsity_allowance`: Stop iteration if the number of zeros in X / the number
    of elements in X > sparsity_allowance.
* `max_iter`: Limit of iterations.
* `max_iter_mp`: Limit of iterations in Matching Pursuit that `ksvd` calls at
    every iteration.
```
"""
function ksvd(Y::AbstractMatrix, n_atoms::Int;
              sparsity_allowance = default_sparsity_allowance,
              max_iter::Int = default_max_iter,
              max_iter_mp::Int = default_max_iter_mp, data_init::Bool, des_avg_err::Float64)

    K = n_atoms
    n, N = size(Y)

    if !(0 <= sparsity_allowance <= 1)
        throw(ArgumentError("`sparsity_allowance` must be in range [0,1]"))
    end

    X = spzeros(K, N)  # just for making X global in this function
    max_n_zeros = ceil(Int, sparsity_allowance * length(X))

    # D is a dictionary matrix that contains atoms for columns.
    if data_init
        D = init_dictionary(n, K)
    else
        D = init_dictionary_data(Y, K)
    end
      # size(D) == (n, K)

    p = Progress(max_iter)
    coeffCutoff=0.01
    maxIP=0.995
    minFracObs=1e-4
    display(string("Beginning Training using ",size(Y,2), " Samples"))

    for i in 1:max_iter
        #X_sparse = matching_pursuit(Y, D, max_iter = max_iter_mp)
        #D, X = ksvd(Y, D, Matrix(X_sparse))
        #display("Sparse Coding (1)")
        X_sparse    = omp(Y, D, max_iter_mp, des_avg_err)

        #display("Atom Updates (2)")
        D, X        = ksvd(Y, D, X_sparse)#omp(Y, D, max_iter_mp, des_avg_err)) #X_sparse)
        avg_err     = norm((Y-D*X)'*(Y-D*X))/size(Y,2)
        ProgressMeter.next!(p; showvalues = [(:iter,i), (:avg_err,avg_err)])

        if avg_err <= des_avg_err
            display("Training Completed Early, Returning D and X")
            println(string("Final iteration: ",i))
            println(string("Average error: ",avg_err))
            return D, X
        end

        #display("Conditioning Dictionary (3)")
        D = condDictionary(D, X, Y, maxIP, coeffCutoff, minFracObs)
    end
    return D, X
end

# JJH 4-27-20 BEGIN
function condDictionary(D::Array{Float64,2}, X::Array{Float64,2}, Y::Array{Float64,2}, maxIP::Float64, coeffCutoff::Float64, minFracObs::Float64)
    # *** replace atoms that:
    # 1) exceed maximum allowed inner product with another atom
    # 2) is used an insufficient number of times for reconstructing observations
    Er = sum((Y-D*X).^2,dims=1) # error in representation
    K= size(D,2)
    N= size(Y,2)
    G = D'*D
    G = G-diagm(diag(G)) # matrix of inner products (diagonal removed)
    for jj = 1:K # run through all atoms
        if maximum(broadcast(abs,G[jj,:])) > maxIP || length(findall(!iszero,broadcast(abs,X[jj,:]).-coeffCutoff))/N  <= minFracObs
            pos = findmax(Er) # sorted indices of obseravtions with highest reconstruction errors
            pos=pos[2] # Extract CartesianIndex
            # replace jj'th atom with normalized data vector with highest reconstruction error
            Er[pos[2]] = 0
            D[:,jj] = Y[:,pos[1]] / norm(Y[:,pos[1]])
        end
    end
    return D
end
# JJH 4-27-20 END


function omp(data::Array{Float64,2}, dictionary::Array{Float64,2}, tau::Int, tolerance::Float64)
    K= size(dictionary,2)
    X_sparse=zeros(K,size(data,2))
    # Solving the sparse codes for each data element (this can be parallelized since dict doesn't change)
    #Bdata=[deepcopy(data) for i=1:Threads.nthreads()]
    Threads.@threads for nn=1:size(data,2)
        #tid = Threads.threadid()
        #xid= Bdata[tid]
        x = deepcopy(data[:,nn])#xid[:,n] # # a single data sample to be represented in terms of atoms in D
        r = deepcopy(x) # residual vector
        D = dictionary # Dictionary
        # Note that the o (optimal) variables are used to avoid an uncommon
        # scenario (that does occur) where a lower sparsity solution may have
        # had lower error than the final solution (with tau non zeros) but
        # wasn't low enough to break out of the coefficient solver via the error
        # tolerance. A litte more memory for significantly better solutions,
        # thanks to CR for the tip (JJH)
        γ       = 0 # this will be the growing coefficient vector
        γₒ      = 0 # this will store whatever the minimum error solution was during computation of the coefficients
        av_err  = 100 # norm of the error vector.
        best_err= 100 # will store lowest error vector norm
        ii      = 1   # while loop index
        DI      = []  # This holds the atoms selected via OMP as its columns (it grows along 2nd dimension)
        DIGI    = []  # Inverse of DI's gram matrix
        DIdag   = []  # PseudoInverse of DI
        I       = []  # set of indices corresponding to atoms selected in reconstruction
        Iₒ      = []  # I think you get the deal with these guys now (best set of indices lul)
        while (length(I)<tau) && (av_err > tolerance)
            k = argmax(broadcast(abs,D'*r))
            dk= D[:,k]
            if ii==1
                I = k
                #display("we made it")
                DI=dk
                DIGI=(DI'*DI)^(-1)
            else
                I = cat(dims=1,I,k)
                rho=DI'*dk
                DI=cat(dims=2,DI,dk)
                ipk=dk'*dk
                DIGI=blockMatrixInv(DIGI,rho,rho,ipk)
            end
            DIdag   = DIGI*DI'
            γ       = DIdag*x
            r       = x-DI*γ
            av_err  = norm(r)
            if av_err<= best_err
                best_err  = av_err
                γₒ        = γ
                Iₒ        = I
            end
            X_sparse[I,nn]=γ
            ii+=1
        end
        if av_err > best_err
            X_sparse[I,nn]= 0*X_sparse[I,nn]
            X_sparse[Iₒ,nn]=γₒ
        end

    end
    return X_sparse
end

function blockMatrixInv(Ai::Array{Float64,2}, B::Array{Float64,1}, C::Array{Float64,1}, D::Float64)
    C=C'
    DCABi= (D-C*Ai*B)^(-1)
    return [Ai+Ai*B*DCABi*C*Ai -Ai*B*DCABi; -DCABi*C*Ai DCABi]
end

function blockMatrixInv(Ai::Float64, B::Float64, C::Float64, D::Float64)
    DCABi= (D-C*Ai*B)^(-1)
    return [Ai+Ai*B*DCABi*C*Ai -Ai*B*DCABi; -DCABi*C*Ai DCABi]
end

end # module
