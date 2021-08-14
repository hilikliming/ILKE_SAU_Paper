module EasyData
export importData, trainTestData

#using MAT
using MLDatasets
using CSV, DataFrames, DelimitedFiles,Random

function importData(dsetname::String, portion::Float64)

    Y,labels=retrieveDataset(dsetname)
    Y,labels=subsampleData(Y,labels,portion)
    return Y,labels
end

function subsampleData(Y::AbstractMatrix, labels::Array{Int64,1}, portion::Float64)
        ssp1 = portion#0.99
        Z1   = Y
        y1   = labels
        # Finding the number of samples that will remain
        ns1     = convert(Int64,floor(size(Z1,2)*ssp1))
        #selecting this many samples from a random perm of indicies
        s1ind   = randperm(size(Z1,2))[1:ns1]
        # Re-sorting our sub-set so that classes will still be chunked together
        s1ind   = sort(s1ind)
        # Selecting our dataset at the desired (sorted) subset of indices.
        Z1      = Z1[:,s1ind]
        y1      = y1[s1ind]
        return Z1,y1
end

function trainTestData(Data::AbstractMatrix,labels::Array{Int64,1},tp::Float64)
    tp1= tp#2/3
    dat1=Data
    labels1=labels

    nTr1=convert(Int64,floor(size(dat1,2)*tp1))
    ind1=randperm(size(dat1,2))

    tr1ind=ind1[1:nTr1]
    te1ind=ind1[nTr1+1:end]

    tr1ind=sort(tr1ind)
    te1ind=sort(te1ind)

    Ztr1=dat1[:,tr1ind]
    ytr1=labels1[tr1ind]
    #Removing Training indices from testing sets
    Zte1=dat1[:,te1ind]
    #Removing Training indices from testing sets labels
    yte1=labels1[te1ind]
    ytr1=vec(convert(Array,ytr1))
    yte1=vec(convert(Array,yte1))
    return Ztr1,ytr1,Zte1,yte1
end

function retrieveDataset(dsetname::String)
    Y=[]
    labels=[]

    display("Datasets Loaded...")
    if dsetname == "TREX13"
        #Open Dataset
        # load full TREX13 set
        Z2  = CSV.read("G:/Code/ILKE_experiments_julia/TREX13Data.csv",DataFrame; header=false)
        y2  = CSV.read("G:/Code/ILKE_experiments_julia/TREX13Labels.csv",DataFrame; header=false)
        Y       = Matrix(Z2)
        labels  = vec(Matrix(y2))
    elseif dsetname=="FRM"
        Y       = CSV.read("G:/Code/ILKE_experiments_julia/FRMData.csv",DataFrame; header=false)
        labels  = CSV.read("G:/Code/ILKE_experiments_julia/FRMLabels.csv",DataFrame; header=false)
        Y       = Matrix(Y)
        labels  = vec(Matrix(labels))
    elseif dsetname == "BAYEX14"
        #Open Dataset
        Z3  = CSV.read("G:/Code/ILKE_experiments_julia/BAYEX14Data.csv",DataFrame; header=false)
        y3  = CSV.read("G:/Code/ILKE_experiments_julia/BAYEX14Labels.csv",DataFrame; header=false)
        Y       = Matrix(Z3)
        labels  = vec(Matrix(y3))
    elseif dsetname == "CLUTTEREX17"
        display("Dataset Currently unsupported")
    elseif dsetname == "MNIST"
        # load full training set
        Z1, y1 = MNIST.traindata()
        Z1=reshape(Z1,size(Z1,1)*size(Z1,2),size(Z1,3))

        # load full test set
        Z2,  y2  = MNIST.testdata()
        Z2=reshape(Z2,size(Z2,1)*size(Z2,2),size(Z2,3))
        Z1= convert(Array{Float64,2},Z1)
        Z2= convert(Array{Float64,2},Z2)

        LiM2 = in.(y1, [0])
        y1[LiM2].=10
        LiM2 = in.(y2, [0])
        y2[LiM2].=10

        Z1=cat(Z1,Z2,dims=2)
        y1=cat(y1,y2,dims=1)

        labels=vec(convert(Array,y1))
        Y=Z1
    elseif dsetname == "USPS"
        Z3      = CSV.read(string(pwd(),"\\backup\\USPSData.csv"),DataFrame; header=false)
        y3      = CSV.read(string(pwd(),"\\backup\\USPSLabels.csv"),DataFrame; header=false)
        Y       = Matrix(Z3)
        labels  = vec(Matrix(y3))
    elseif dsetname == "CIFAR10"
        # load full training set
        Z1=reshape(CIFAR10.traintensor(Float32), 3072,50000)#CIFAR10.convert2features(CIFAR10.traintensor(Float32))
        y1=CIFAR10.trainlabels()

        Z2=reshape(CIFAR10.testtensor(Float32), 3072,10000)#CIFAR10.convert2features(CIFAR10.testtensor(Float32))
        y2=CIFAR10.testlabels()

        LiM2 = in.(y1, [0])
        y1[LiM2].=10
        LiM2 = in.(y2, [0])
        y2[LiM2].=10

        Z1=cat(Z1,Z2,dims=2)
        y1=cat(y1,y2,dims=1)

        labels=vec(convert(Array,y1))
        Y=Z1
    else
        display("Invalid Dataset Selected")
    end

    return Y, labels

end


end # module
