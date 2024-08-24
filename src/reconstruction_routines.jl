function read_model_data(;which_files=nothing,       # indices of files used for training     ; e.g. 1:10, default all available
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

function prepare_model(model_files, ref_file, outline_mask_file, bedm_file, r, use_arpack)
    # get I_no_ocean
    h_ref        = NCDataset(ref_file)["Band1"][:]
    h_ref        = replace_missing(h_ref, 0.0)
    outline_mask = ncread(outline_mask_file, "Band1")
    grimp_mask   = ncread(bedm_file, "mask")
    I_no_ocean   = findall( (vec(grimp_mask) .!= 1) .&& vec(outline_mask) .> 0.0)

    # load model data and calculate difference to reference DEM
    Data_ice  = read_model_data(;model_files,I_no_ocean)

    # calculate difference to reference DEM
    data_ref  = h_ref[I_no_ocean]
    Data_ice .= data_ref .- Data_ice

    # centering model data
    data_mean  = mean(Data_ice, dims=2)
    Data_ice  .= Data_ice .- data_mean

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
    return UΣ, I_no_ocean, data_mean, data_ref, Σ
end

function prepare_obs(obs_aero_file, ref_file, atm_dh_grid_file, I_no_ocean, data_mean, fig_dir)
    # aerodem
    h_aero                     = NCDataset(obs_aero_file)["Band1"][:]
    idx_aero                   = findall(.!ismissing.(h_aero[I_no_ocean]))
    h_aero                     = replace_missing(h_aero, 0.0)
    h_ref                      = NCDataset(ref_file)["Band1"][:]
    h_ref                      = replace_missing(h_ref, 0.0)
    obs                        = zeros(size(h_aero))
    obs[I_no_ocean[idx_aero]] .= h_ref[I_no_ocean[idx_aero]] .- h_aero[I_no_ocean[idx_aero]]

    # atm
    dh_atm                    = NCDataset(atm_dh_grid_file)["Band1"][:]
    aero_dilated              = dilate(h_aero, r=10)
    idx_atm                   = findall(.!ismissing.(dh_atm[I_no_ocean]) .&& aero_dilated[I_no_ocean] .== 0.0)
    to_del                    = findall(abs.(dh_atm[I_no_ocean[idx_atm]]) .> 5 * mad(dh_atm[I_no_ocean[idx_atm]]))
    deleteat!(idx_atm, to_del)    # there are some extreme dh atm values, better to filter them out
    obs[I_no_ocean[idx_atm]] .= dh_atm[I_no_ocean[idx_atm]]

    # obtain I_obs
    I_obs      = findall(obs[I_no_ocean] .!= 0.0)

    # center observations
    x_data = obs[I_no_ocean][I_obs] .- data_mean[I_obs]

    # save matrix of observations as nc and plot
    gr = Int(diff(NCDataset(obs_aero_file)["x"][1:2])[1])
    save_netcdf(joinpath(dirname(fig_dir),"obs_all_gr$(gr).nc"), obs_aero_file, [obs], ["dh"], Dict("dh"=>Dict{String,Any}()))
    Plots.heatmap(obs', clims=(-100,100), cmap=:bwr)
    Plots.savefig(joinpath(fig_dir, "obs_all_g$(gr).png"))
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

function SVD_reconstruction(λ::Real, r::Int, gr::Int, outline_shp_file, model_files::Vector{String}, use_arpack=false)
    # define output paths
    main_output_dir = "output/SVD_reconstruction/"
    fig_dir = joinpath(main_output_dir, "figures")
    mkpath(main_output_dir)
    mkpath(fig_dir)

    # get filenames
    bedmachine_original, bedm_file  = create_bedmachine_grid(gr)
    reference_file_g150, ref_file   = create_grimpv2(gr, bedmachine_original)
    aerodem_g150, obs_aero_file     = create_aerodem(gr, outline_shp_file, bedmachine_original, reference_file_g150)
    atm_dh_grid_file                = get_atm_grid(gr, ref_file, bedm_file)
    outline_mask_file               = create_outline_mask(gr, outline_shp_file, aerodem_g150)

    UΣ, I_no_ocean, data_mean, data_ref, _ = prepare_model(model_files, ref_file, outline_mask_file, bedm_file, r, use_arpack) # read in model data and take svd to derive "eigen ice sheets"
    x_data, I_obs                          = prepare_obs(obs_aero_file, ref_file, atm_dh_grid_file, I_no_ocean, data_mean, fig_dir)
    r                                      = min(size(UΣ,2), r)                                                      # truncation of SVD cannot be higher than the second dimension of U*Σ
    x_rec                                  = solve_optim(UΣ, I_obs, r, λ, x_data)                                    # derive analytical solution of regularized least squares

    # calculate error and print
    nx, ny                  = size(NCDataset(obs_aero_file)["Band1"])
    dif                     = zeros(F, nx,ny)
    dif[I_no_ocean[I_obs]] .= x_rec[I_obs] .- x_data
    err_mean                = mean(abs.(dif[I_no_ocean[I_obs]]))
    @printf("Mean absolute error: %1.1f m\n", err_mean)

    # retrieve matrix of reconstructed DEM
    dem_rec                 = zeros(F, nx,ny)
    dem_rec[I_no_ocean]     = data_ref .- (x_rec .+ data_mean)
    dem_rec[dem_rec .== 0] .= no_data_value

    # save as nc file
    println("Saving file..")
    logλ        = Int(round(log(10, λ)))
    filename    = joinpath(main_output_dir,"rec_lambda_1e$logλ"*"_g$gr"*"_r$r.nc")
    layername   = "surface"
    attributes  = Dict(layername => Dict{String, Any}("long_name" => "ice surface elevation",
                                                      "standard_name" => "surface_altitude",
                                                      "units" => "m")
                        )
    save_netcdf(filename, obs_aero_file, [dem_rec], [layername], attributes)

    # plot and save difference between reconstruction and observations
    save_netcdf(joinpath(main_output_dir, "dif_lambda_1e$logλ"*"_g$gr"*"_r$r.nc"), obs_aero_file, [dif], ["dif"], Dict("dif" => Dict{String,Any}()))
    Plots.heatmap(reshape(dif,nx,ny)', cmap=:bwr, clims=(-200,200), cbar_title="[m]", title="reconstructed - observations", size=(700,900))
    Plots.savefig(joinpath(fig_dir, "dif_lambda_1e$logλ"*"_g$gr"*"_r$r.png"))

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
