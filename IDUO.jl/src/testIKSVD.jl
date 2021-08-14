include("IDUO.jl")
include("IKSVD.jl")
using Main.IDUO
using Main.IKSVD
using MLDatasets,MLBase
using CSV, DataFrames, DelimitedFiles, Random, LinearAlgebra
using Colors, Plots
using Gadfly

subsampleData= true
norm_col_data= true

## Opening MNIST and USPS datasets
Z1, y1 = MNIST.traindata()
Z1=reshape(Z1,size(Z1,1)*size(Z1,2),size(Z1,3))

# load full test set
Z2,  y2  = MNIST.testdata()
Z2=reshape(Z2,size(Z2,1)*size(Z2,2),size(Z2,3))
Z1= convert(Array{Float64,2},Z1)
Z2= convert(Array{Float64,2},Z2)

# Represent the 0 digits as class 10 !!
LiM2 = in.(y1, [0])
y1[LiM2].=10
LiM2 = in.(y2, [0])
y2[LiM2].=10

#f1=CSV.open("USPSData.csv")
#f2=CSV.open("USPSLabels.csv")
f1 = open("G:/Code/IDUO.jl/src/USPSData.csv")
f2 = open("G:/Code/IDUO.jl/src/USPSLabels.csv")
Z3  = CSV.read(f1; header=false)
y3  = CSV.read(f2; header=false)
close(f1)
close(f2)

Z3  = convert(Array{Float64,2},Z3)
y3  = vec(convert(Array,y3))

Z1  = cat(Z1,Z2,dims=2) # Grouping all of MNIST together, we will break into training and testing later.
y1  = cat(y1,y2,dims=1) # Grouping MNIST labels together

Z2 = Z3
y2 = y3
display("Datasets Loaded...")
## Subsampling the datasets since we are just proving a concept and training can take a while
if subsampleData
    ssp1=0.25#0.99
    ssp2=0.5#0.99#0.25#
    #ssp3=0.99

    ns1=convert(Int64,floor(size(Z1,2)*ssp1))
    ns2=convert(Int64,floor(size(Z2,2)*ssp2))
    #ns3=convert(Int64,floor(size(Z3,2)*ssp3))

    s1ind=randperm(size(Z1,2))[1:ns1]
    s2ind=randperm(size(Z2,2))[1:ns2]
    #s3ind=randperm(size(Z3,2))[1:ns3]

    s1ind=sort(s1ind)
    s2ind=sort(s2ind)
    #s3ind=sort(s3ind)

    Z1=Z1[:,s1ind]
    y1=y1[s1ind]
    Z2=Z2[:,s2ind]
    y2=y2[s2ind]
end

if norm_col_data
    for ii = 1:size(Z1,2)
        Z1[:,ii]=normalize(Z1[:,ii])
    end
    for ii = 1:size(Z2 ,2)
        Z2[:,ii]=normalize(Z2[:,ii])
    end
end


## Setting aside training and testing partitions...
# Training percentages for Z1 and Z2
tp1= 6/7#2/3
tp2= 0.1

nTr1=convert(Int64,floor(size(Z1,2)*tp1))
nTr2=convert(Int64,floor(size(Z2,2)*tp2))

ind1=randperm(size(Z1,2))
ind2=randperm(size(Z2,2))

# Take the first nTr indices for training and use remaining for testing.
tr1ind=ind1[1:nTr1]
tr2ind=ind2[1:nTr2]
te1ind=ind1[nTr1+1:end]
te2ind=ind2[nTr2+1:end]

# Sorting again since our labels were sorted to begin with
tr1ind=sort(tr1ind)
tr2ind=sort(tr2ind)
te1ind=sort(te1ind)
te2ind=sort(te2ind)

# Selecting training samples using training indices
Ztr1=Z1[:,tr1ind]
Ztr2=Z2[:,tr2ind]
ytr1=y1[tr1ind]
ytr2=y2[tr2ind]

#Selecting Testing Samples
Zte1=Z1[:,te1ind]
Zte2=Z2[:,te2ind]

#Selecting Testing sample labels
yte1=y1[te1ind]
yte2=y2[te2ind]

# We don't need to use these anymore and they are kinda big
Z1=0
Z2=0

ytr1=vec(convert(Array,ytr1))
ytr2=vec(convert(Array,ytr2))

yte1=vec(convert(Array,yte1))
yte2=vec(convert(Array,yte2))

uni_m=unique(ytr1)
uni_m=sort(uni_m)

tau         = 10
max_iter    = 20
K           = 500
des_avg_err = 1e-5
SA          = tau/K
data_init   = true

Ds=zeros(length(uni_m),size(Ztr1,1),K)
for ii in uni_m
        LiM = in.(ytr1, [ii])
        println(string("Training Dictionary No. ",string(ii),"\n"))
        @time D, X = IKSVD.ksvd(Ztr1[:,LiM], K, max_iter=max_iter, max_iter_mp = tau, sparsity_allowance = SA,data_init=data_init,des_avg_err=des_avg_err)
        iint=convert(Int64,ii)
        Ds[iint,:,:]=D
end
display("Training Complete")
display("Testing SRC Classification with Baseline Dictionaries...")

dset=1
if dset ==1
    Xte=Zte1
    yte=yte1
else
    Xte=Zte2
    yte=yte2
end


stats=zeros(length(uni_m),size(Xte,2))
# Generating OMP-MSC Statistics with each of the dictionaries
for class in uni_m
    class_i=convert(Int64,class)
    D = Ds[class_i,:,:]
    println(string("Computing OMP-MSC Stats for class ", class))
    XOMP=IKSVD.omp(Xte,D,tau,des_avg_err)
    stats[class_i,:]= [(Xte[:,ii]-D*XOMP[:,ii])'*(Xte[:,ii]-D*XOMP[:,ii]) for ii in 1:size(Xte,2)]#diag((Xte-D*XOMP)'*(Xte-D*XOMP))
end

decs=argmin(stats,dims=1)
decs=[ i[1] for i in decs ] #or map(i->i[2], decs)
decs=vec(decs)
yte=convert(Vector{Int64},yte)

display(string("Dataset No. ",string(dset)," Performance:"))
display(sum(yte.==decs)/length(yte))

C1=confusmat(length(uni_m),yte,decs)
C1=convert(Array{Int64},C1)
ss=sum(C1,dims=2)
C1=C1./ss
display(C1)

display("Updating Dictionaries with a Limited Number of Samples from USPS")
K1=3
Ds2=zeros(length(uni_m),size(Ztr1,1),K+K1)
for ii in uni_m
        LiM = in.(ytr2, [ii])
        class_i=convert(Int64,ii)
        Dold = Ds[class_i,:,:]
        println(string("Training Dictionary No. ",string(ii),"\n"))
        # if use_iduo
        #@time D, X = IDUO.iduo(Ztr2[:,LiM], Dold, K1, max_iter=max_iter, max_iter_mp = tau, data_init=data_init, des_avg_err=des_avg_err)
        # else
        @time D, X = IKSVD.iksvd(Ztr2[:,LiM], Dold, K1, max_iter=max_iter, max_iter_mp = tau, sparsity_allowance = SA, data_init=data_init, des_avg_err=des_avg_err)
        # end
        iint=convert(Int64,ii)
        Ds2[iint,:,:]=D
    end
    display("Training Complete.")
    display("Re-Testing USPS Test set with Updated Dictionaries")

for class in uni_m
    class_i=convert(Int64,class)

    D = Ds2[class_i,:,:]
    println(string("Computing OMP-MSC Stats for class ", class))
    XOMP=IKSVD.omp(Xte,D,tau,des_avg_err)
    stats[class_i,:]= [(Xte[:,ii]-D*XOMP[:,ii])'*(Xte[:,ii]-D*XOMP[:,ii]) for ii in 1:size(Xte,2)]#diag((Xte-D*XOMP)'*(Xte-D*XOMP))
end

decs=argmin(stats,dims=1)
decs=[ i[1] for i in decs ] #or map(i->i[2], decs)
decs=vec(decs)
yte=convert(Vector{Int64},yte)
display(string("Dataset No. ",string(dset)," Performance:"))
display(sum(yte.==decs)/length(yte))

C2=confusmat(length(uni_m),yte,decs)
C2=convert(Array{Int64},C2)
ss=sum(C2,dims=2)
C2=C2./ss
display(C2)

## Visualize the learned atoms
#D1=reshape(Ds,size(Ds,2),size(Ds,3),size(Ds,1))
#D1=reshape(D1,size(D1,1),size(D1,2)*size(D1,3))
# atoms_grid=zeros(2800,280)
# for atom = 1:size(D1,2)
#     sx= convert(Int64,mod(atom,10))
#     sy= convert(Int64,floor(atom/100))
#     atom_img=reshape(D1[:,atom],28,28)
no_atoms=10#size(Ds,3)
atoms_grid=zeros(280,28*no_atoms)
for mm in 1:size(Ds,1)
    for nn in 1:no_atoms
        atom_img=reshape(Ds[mm,:,nn],28,28)'
        atoms_grid[(mm-1)*28+1:mm*28,(nn-1)*28+1:nn*28]=atom_img
    end
end

# end

plot(Gray.(atoms_grid),size= 4 .*size(atoms_grid)[end:-1:1])
#Gadfly.plot(atoms_grid)
