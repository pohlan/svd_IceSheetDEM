using Flux, Optim, Plots, TSVD, IterTools, Statistics, LinearAlgebra, Compat, Glob, NetCDF

include("read_in.jl")
include("do_training.jl")

Data, I, I_inv, R, nx, ny = make_training_data(rtrain=1:1,tsteps=1:10,t_y=10)

# center the data
Data_mean = mean(Data, dims=1)
Data_centr = Data .- Data_mean

qs = 4:10
λs = [1e4] #[1e2 1e3 1e4 1e5 1e6]
losses_e = zeros(length(qs), length(λs))
dhs      = zeros(length(qs), length(λs))
Ms       = zeros(length(qs), length(λs),nx*ny)
###########################################
for (i,q) in enumerate(qs)
    # compute the SVD
    U, S, V = tsvd(Data_centr, q)
    US = U * diagm(S)

    # prepare training
    x_train = US[repeat(I, 1, q)]
    n_obs, _ = size(x_train)
    R_centr = R .- Data_mean         # center R as well
    y_train = reshape(R_centr[I], n_obs, 1)

    for (j,λ) in enumerate(λs)
        println("Training for λ = $λ, q = $q ...")

        # do training
        M, loss_e = optim_training(;x_train,y_train,US,λ)
        M = M .+ Data_mean
        M[findall(x->x<0, M)] .= 0

        # save error
        losses_e[i,j] = loss_e
        dhs[i,j] = Flux.Losses.mse(M[I_inv],R[I_inv])
        Ms[i,j,:] = M
    end
end
###########################################


# filename = "dem_R.nc"
# jldsave(filename;M)



# "true" final field at the margin -> blend between data and minimization result ?
# smoothing where the interior touches I?


