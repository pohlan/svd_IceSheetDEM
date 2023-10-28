# function to read in model data
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
    elseif all(length(tsteps) .<= nts)
        nttot = length(tsteps)*nf
    else
        error("Time steps out of bound for at least one file.")
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
        ts = isnothing(tsteps) ? (1:size(d, 3)) : tsteps
        nt_out = length(ts)
        data = reshape(d, ny*nx, nt_out)
        @views Data[:, ntcount+1 : ntcount+nt_out] = data[I_no_ocean,ts]
        ntcount += nt_out
    end
    return Data, nx, ny
end

"""
Get indices of cells with observations
"""
function get_indices(obs::Matrix{T}, mask_path::String, mask_name="Band1") where T<:Real
    # load imbie mask
    imbie_mask    = ncread(mask_path, mask_name)
    no_ocean_mask = findall((vec(obs) .> 0.0) .|| (vec(imbie_mask) .== 1))
    # get indices where there is data and ice, with respect to ice_mask
    R      = obs[no_ocean_mask]  # vector
    I_obs         = findall(R .> 0.0)
    return no_ocean_mask, I_obs
end
