function read_model_data(;F::DataType=Float32,       # Float32 or Float64
                          which_files=nothing,       # indices of files used for training     ; e.g. 1:10, default all available
                          tsteps=nothing,            # indices of time steps used for training; e.g.  ""      ""
                          model_files,
                          I_no_ocean)
    println("Reading in model data...")
    # determine indices for files
    if which_files === nothing
        which_files = 1:length(model_files)
    elseif which_files[end] > length(model_files)
        error("Number of files out of bound.")
    end
    files_out  = model_files[which_files]
    nf         = length(files_out)

    # determine total number of time steps
    nts = []
    for f in files_out
        ds   = NCDataset(f)
        nt_f = size(ds["usurf"], 3)
        push!(nts, nt_f)
        close(ds)
    end
    if isnothing(tsteps)
        nttot = sum(nts)
    else
        nttot = sum(min.(nts, length(tsteps)))
    end

    # determine number of cells in x and y direction
    ds = NCDataset(files_out[1])
    nx, ny = size(ds["usurf"])[1:2]
    close(ds)
    # build data matrix
    Data = zeros(F, length(I_no_ocean), nttot)
    ntcount = 0
    @showprogress for (k, file) in enumerate(files_out)
    d = ncread(file, "usurf")
    if isnothing(tsteps)
        ts = 1:size(d,3)
    elseif minimum(tsteps) > size(d,3)
        continue
    else
        ts = tsteps[1]:min(tsteps[end], size(d, 3))
    end
    nt_out = length(ts)
    data = reshape(d[:,:,ts], ny*nx, nt_out)
    @views Data[:, ntcount+1 : ntcount+nt_out] = data[I_no_ocean,:]
    ntcount += nt_out
    end
    return Data
end

function prepare_model(model_files, imbie_path, bedm_path, r, F, use_arpack)
    # get I_no_ocean
    imbie_mask = ncread(imbie_path, "Band1")
    grimp_mask = ncread(bedm_path, "mask")
    I_no_ocean = findall( (vec(grimp_mask) .!= 1) .&& vec(imbie_mask) .> 0.0)

    # load model data
    Data_ice = read_model_data(;F,model_files,I_no_ocean)
    # centering model data
    Data_mean  = mean(Data_ice, dims=2)
    Data_ice  .= Data_ice .- Data_mean

    # compute SVD
    if use_arpack
        nsv = min(r, size(Data_ice, 2)-1) # actual truncation is later, but takes too long if r is unnecessarily high here
        B, _ = svds(Data_ice; nsv)
        U, Σ, V = B
    else
        U, Σ, V = svd(Data_ice)
    end

    # prepare least square fit problem
    UΣ            = U*diagm(Σ)
    return UΣ, I_no_ocean, Data_mean, Σ
end

function prepare_obs(obs_file, I_no_ocean, Data_mean)
    # load observations
    obs = ncread(obs_file, "Band1")
    # obtain I_obs
    I_obs      = findall(obs[I_no_ocean] .> 0.0)
    # center observations
    x_data = obs[I_no_ocean][I_obs] .- Data_mean[I_obs]
    return x_data, I_obs
end

function solve_optim(UΣ::Matrix{T}, I_obs::Vector{Int}, r::Int, λ::Real, x_data) where T <: Real
    @views A      = UΣ[I_obs,1:r]
    U_A, Σ_A, V_A = svd(A)
    D             = transpose(diagm(Σ_A))*diagm(Σ_A) + λ*I
    v_rec         = V_A * D^(-1) * transpose(diagm(Σ_A)) * transpose(U_A) * x_data
    x_rec         = UΣ[:,1:r]*v_rec
    return x_rec
end

function do_reconstruction(F::DataType, λ::Real, r::Int, gr::Int, imbie_mask::String, bedm_file::String, model_files::Vector{String}, obs_file::String, do_figures=false, use_arpack=false)
    UΣ, I_no_ocean, Data_mean, _ = prepare_model(model_files, imbie_mask, bedm_file, r, F, use_arpack) # read in model data and take svd to derive "eigen ice sheets"
    x_data, I_obs                = prepare_obs(obs_file, I_no_ocean, Data_mean)
    r                            = min(size(UΣ,2), r)                                                    # truncation of SVD cannot be higher than the second dimension of U*Σ
    x_rec                        = solve_optim(UΣ, I_obs, r, λ, x_data)                                  # derive analytical solution of regularized least squares

    # calculate error and print
    nx, ny                  = values(NCDataset(obs_file).dim)
    dif                     = zeros(F, nx,ny)
    dif[I_no_ocean[I_obs]] .= x_rec[I_obs] .- x_data
    err_mean                = mean(abs.(dif[I_no_ocean[I_obs]]))
    @printf("Mean absolute error: %1.1f m\n", err_mean)

    # retrieve matrix of reconstructed DEM
    dem_rec             = zeros(F, nx,ny)
    dem_rec[I_no_ocean] = x_rec .+ Data_mean
    # smoothing
    dem_rec = mapwindow(median, dem_rec, (5,5))
    dem_rec[dem_rec .== 0] .= no_data_value

    # save as nc file
    mkpath("output/")
    println("Saving file..")
    logλ        = Int(round(log(10, λ)))
    filename    = "output/rec_lambda_1e$logλ"*"_g$gr"*"_r$r.nc"
    layername   = "surface"
    attributes  = Dict(layername => Dict{String, Any}("long_name" => "ice surface elevation",
                                                      "standard_name" => "surface_altitude",
                                                      "units" => "m")
                        )
    save_netcdf(filename, obs_file, [dem_rec], [layername], attributes)
    # plot and save difference between reconstruction and observations
    if do_figures
        save_netcdf("output/dif.nc", obs_file, [dif], ["dif"], Dict("dif" => Dict()))
        Plots.heatmap(reshape(dif,nx,ny)', cmap=:bwr, clims=(-200,200), cbar_title="[m]", title="reconstructed - observations", size=(700,900))
        Plots.savefig(filename[1:end-3]*".png")
    end

    return filename
end

function create_reconstructed_bedmachine(rec_file, bedmachine_file)
    # load datasets
    surfaceDEM = ncread(rec_file, "surface")
    bedDEM     = ncread(bedmachine_file, "bed")
    bedm_mask  = ncread(bedmachine_file, "mask")
    ice_mask   = (surfaceDEM .!= no_data_value) .&& (surfaceDEM .> bedDEM)

    # retrieve grid size
    x          = ncread(rec_file, "x")
    gr         = Int(x[2] - x[1])

    # calculate floating mask
    ρw            = 1030
    ρi            = 917
    Pw            = - ρw * bedDEM
    Pi            = ρi * (surfaceDEM - bedDEM)
    floating_mask = (Pw .> Pi) .&& (ice_mask)  # floating where water pressure > ice pressure at the bed

    # calculate mask
    new_mask = ones(eltype(bedm_mask), size(bedm_mask))
    new_mask[bedm_mask .== 0] .= 0 # ocean
    new_mask[ice_mask] .= 2
    new_mask[floating_mask]   .= 3

    # make sure DEM is zero everywhere on the ocean and equal to bed DEM on bedrock
    surfaceDEM[new_mask .== 0] .= no_data_value
    surfaceDEM[new_mask .== 1] .= bedDEM[new_mask .== 1]

    # calculate ice thickness
    h_ice = zeros(size(surfaceDEM)) .+ no_data_value
    h_ice[ice_mask] .= surfaceDEM[ice_mask] - bedDEM[ice_mask]
    h_ice[floating_mask] .= surfaceDEM[floating_mask] ./  (1-ρi/ρw)

    # save to netcdf file
    dest        = "output/bedmachine1980_reconstructed_g$(gr).nc"
    layers      = [surfaceDEM, bedDEM, h_ice, new_mask]
    layernames  = ["surface", "bed", "thickness", "mask"]
    template    = NCDataset(bedmachine_file)
    attributes  = get_attr(template, layernames)
    # overwrite some attributes
    sources_rec = Dict("surface"   => "svd reconstruction",
                       "bed"       => "Bedmachine-v5: Morlighem et al. (2022). IceBridge BedMachine Greenland, Version 5. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/GMEVBWFLWA7X; projected on new grid with gdalwarp",
                       "thickness" => "computed from surface and bed",
                       "mask"      => "bedrock from Morlighem et al. (2022); ice, floating and ocean computed from surface and bed elevation"
                       )
    for l in layernames
        attributes[l]["source"] = sources_rec[l]
    end
    attributes["mask"]["long_name"] = "mask (0 = ocean, 1 = ice-free land, 2 = grounded ice, 3 = floating ice)"
    save_netcdf(dest, bedmachine_file, layers, layernames, attributes)

    return dest
end
