function make_geotable(Z_data, xs, ys)
    table    = (; Z=Float32.(Z_data))
    coords   = [(xi,yi) for (xi,yi) in zip(xs, ys) ]
    return georef(table,coords)
end

"""
Binning 1D or 2D based on equal sample size per bin
"""
function bin_equal_sample_size(x, y, n_samples)   # 1D
    nx = length(x)
    n_bins = ceil(Int, nx / n_samples)
    if nx % n_samples < 100
        n_bins -= 1
    end
    p = sortperm(x)
    out = Vector{Vector{eltype(y)}}(undef,n_bins)
    bin_centers = zeros(eltype(x), n_bins)
    for b in 1:n_bins
        if b == n_bins
            i = (b-1)*n_samples+1:nx
        else
            i = (b-1)*n_samples+1:b*n_samples
        end
        bin_centers[b] = 0.5*(sort(x)[i[1]]+sort(x)[i[end]])
        out[b] = y[p[i]]
    end
    return out, bin_centers
end
function bin_equal_sample_size(x1, x2, y, n_bins_1, n_bins_2) # 2D
    @assert length(x1) == length(x2) == length(y)
    nx = length(x1)
    nsampl1 = ceil(Int, nx / n_bins_1)
    nsampl2 = ceil(Int, nx / n_bins_2)
    if nx % nsampl1 < 500
        n_bins_1 -= 1
    end
    if nx % nsampl2 < 500
        n_bins_2 -= 1
    end
    p1 = sortperm(x1)
    p2 = sortperm(x2)
    bin_edges_1 = x1[p1[[1:nsampl1:(n_bins_1-1)*nsampl1+1;nx]]]
    bin_edges_2 = x2[p2[[1:nsampl2:(n_bins_2-1)*nsampl2+1;nx]]]
    out = Array{Vector{eltype(y)}}(undef,n_bins_1,n_bins_2)
    for b1 in 1:n_bins_1
        for b2 in 1:n_bins_2
            i = findall(bin_edges_1[b1] .< x1 .< bin_edges_1[b1+1] .&& bin_edges_2[b2] .< x2 .< bin_edges_2[b2+1])
            out[b1,b2] = y[i]
        end
    end
    bin_centers_1 = bin_edges_1[1:end-1] .+ 0.5*diff(bin_edges_1)
    bin_centers_2 = bin_edges_2[1:end-1] .+ 0.5*diff(bin_edges_2)
    return out, bin_centers_1, bin_centers_2
end


"""
    filter_per_bin

Input:
- idx_binned
-
"""
function filter_per_bin!(y_binned; cutoff=5.0)
    println("Removing outliers per bin..")
    n_before = sum(length.(y_binned))
    is_deleted = [Int[] for i in y_binned]
    @showprogress for (i, (y_bin)) in enumerate(y_binned)
        if !isempty(y_bin)
            nmad_ = StatsBase.mad(y_bin, normalize=true)
            i_to_delete = findall(abs.(y_bin) .> cutoff*nmad_)
            sort!(unique!(i_to_delete))  # indices must be unique and sorted for keepat!
            deleteat!(y_bin,  i_to_delete)
            is_deleted[i] = i_to_delete
        end
    end
    n_after  = sum(length.(y_binned))
    perc_deleted = (n_before-n_after)*100 / n_before
    @printf("%.2f %% of points removed as outliers.\n", perc_deleted)
    return is_deleted
end

function replace_missing(A,c)
    A[ismissing.(A)] .= c
    return A
end

function get_ATM_df(fname, dhdt0, x, y, df_aero; mindist=5e4, I_no_ocean)
    df_atm = CSV.read(fname, DataFrame)
    df_atm[!,:source] .= :atm
    dhdt = replace_missing(dhdt0, 0)
    sort!(unique!(x))   # necessary for interpolation
    sort!(unique!(y))
    itp = interpolate((x, y), dhdt, Gridded(Linear()))
    df_atm[!,:dhdt] = itp.(df_atm.x, df_atm.y)
    max_dhdt = std(dhdt[dhdt .!= 0])
    function choose_atm(x_, y_, dhdt_)::Bool
        ix    = findmin(abs.(x_ .- x))[2]
        iy    = findmin(abs.(y_ .- y))[2]
        iglob = get_global_i(ix, iy, length(x))
        # filter values outside the ice sheet
        is_in_icesheet = iglob ∈ I_no_ocean
        # filter values overlapping or close to aerodem
        dist_to_aero = minimum(pairwise(Distances.euclidean, [x_ y_], [df_aero.x df_aero.y], dims=1)[:])
        is_above_mindist = dist_to_aero > mindist
        # filter values with high absolute dhdt
        has_low_dhdt = 0.0 < abs(dhdt_) < max_dhdt
        return is_in_icesheet & is_above_mindist & has_low_dhdt
    end
    println("selecting flightline values...")
    filter!([:x,:y,:dhdt] => choose_atm, df_atm)
    # already remove some outliers here, improves standardization
    atm_to_delete = findall(abs.(df_atm.dh) .> 5 .* mad(df_atm.dh))
    deleteat!(df_atm, atm_to_delete)
    return df_atm
end

function get_aerodem_df(aero, ref, dhdt0, x, y, idx_aero)
    dhdt = replace_missing(dhdt0, 0)
    df_aero  = DataFrame(:x       => x[get_ix.(idx_aero, length(x))],
                         :y       => y[get_iy.(idx_aero, length(x))],
                         :h_ref   => ref[idx_aero],
                         :dhdt    => dhdt[idx_aero],
                         :dh      => ref[idx_aero] - aero[idx_aero],
                         :idx     => idx_aero,
                         :source .=> :aerodem )
    return df_aero
end

""""
    remove_small_bins!()
Remove bins where number of samples is below a threshold.
Input
    - min_n_sample: number of samples that a bin is required to have
"""
function remove_small_bins!(A_binned::Vector; min_n_sample=80)  # 1D
    i_rm  = findall(length.(A_binned[1]) .< min_n_sample)
    for A in A_binned
        deleteat!(A, i_rm)
    end
    @printf("Removed %d bins, %d bins left.\n", length(i_rm), length(A_binned[1]))
    return
end
function remove_small_bins(bin_centers_1::Vector, bin_centers_2::Vector, A_binned::Matrix; min_n_sample=80) # 2D
    while any(length.(A_binned) .< min_n_sample)
        i_rm               = findall(length.(A_binned) .< min_n_sample)
        ix_rm              = [first(i_rm[i].I) for i in eachindex(i_rm)]
        iy_rm              = [last(i_rm[i].I) for i in eachindex(i_rm)]
        x_maxcount, ix_max = findmax(countmap(ix_rm))
        y_maxcount, iy_max = findmax(countmap(iy_rm))
        if x_maxcount > y_maxcount || (x_maxcount == y_maxcount && size(A_binned,1)>size(A_binned,2))
            idx_x = Vector(1:size(A_binned, 1))
            deleteat!(idx_x, ix_max)
            A_binned = A_binned[idx_x, :]
            bin_centers_1 = bin_centers_1[idx_x]
            @printf("Removed one row, %d rows left.\n", length(idx_x))
        else
            idx_y = Vector(1:size(A_binned, 2))
            deleteat!(idx_y, iy_max)
            A_binned = A_binned[:, idx_y]
            bin_centers_2 = bin_centers_2[idx_y]
            @printf("Removed one column, %d columns left.\n", length(idx_y))
        end
    end
    @assert all(length.(A_binned) .>= min_n_sample)
    return bin_centers_1, bin_centers_2, A_binned
end

function standardizing_2D(df::DataFrame, bfield1::Symbol, bfield2::Symbol; nbins1, nbins2, min_n_sample=100, fig_path="")
    bin_field_1 = df[!,bfield1]
    bin_field_2 = df[!,bfield2]
    y_binned, bin_centers_1, bin_centers_2 = bin_equal_sample_size(bin_field_1, bin_field_2, Float64.(df.dh), nbins1, nbins2)
    filter_per_bin!(y_binned)
    bin_centers_1, bin_centers_2, y_binned = remove_small_bins(bin_centers_1, bin_centers_2, y_binned; min_n_sample)
    # variance
    nmads       = mad.(y_binned, normalize=true)
    itp_mad_lin = interpolate((bin_centers_1, bin_centers_2), nmads, Gridded(Linear()))
    itp_mad_lin = extrapolate(itp_mad_lin, Interpolations.Flat())
    # bias
    meds        = median.(y_binned)
    itp_med_lin = interpolate((bin_centers_1, bin_centers_2), meds, Gridded(Linear()))
    itp_med_lin = extrapolate(itp_med_lin, Interpolations.Flat())
    # standardize
    df.dh_detrend   = (df.dh .- itp_med_lin.(bin_field_1,bin_field_2)) ./ itp_mad_lin.(bin_field_1,bin_field_2)
    # remove outliers again after standardizing
    all_to_delete = findall(abs.(df.dh_detrend) .> 4 .* mad(df.dh_detrend))
    deleteat!(df, all_to_delete)
    # make sure it's truly centered around zero and has std=1 exactly
    std_y          = std(df.dh_detrend)
    mean_y         = mean(df.dh_detrend)
    df.dh_detrend  = (df.dh_detrend .- mean_y) ./ std_y
    if !isempty(fig_path)
        # nmads
        Plots.heatmap(bin_centers_1, bin_centers_2, nmads')
        Plots.savefig(joinpath(fig_path, "nmads_2Dbinning.png"))
        # bias
        Plots.heatmap(bin_centers_1, bin_centers_2, meds')
        Plots.savefig(joinpath(fig_path, "medians_2Dbinning.png"))
        # histograms
        Plots.histogram(df.dh_detrend, label="Standardized observations", xlims=(-10,10), normalize=:pdf, nbins=1000, wsize=(600,500), linecolor=nothing)
        Plots.plot!(Normal(), lw=1, label="Normal distribution", color="black")
        Plots.savefig(joinpath(fig_path,"histogram_standardization.png"))
        # qqplot
        Plots.plot(
            StatsPlots.qqplot(StatsPlots.Normal(), df.dh_detrend, title="standardized with binning", ylims=(-8,8)),
            StatsPlots.qqplot(StatsPlots.Normal(), (df.dh .- mean(df.dh))./std(df.dh), title="standardized without binning", ylims=(-8,8))
        )
        Plots.savefig(joinpath(fig_path,"qqplot.png"))
        # nmad interpolation
        x1 = range(bin_centers_1[1], bin_centers_1[end], length=10000)
        x2 = range(bin_centers_2[1], bin_centers_2[end], length=1000)
        Plots.heatmap(x1, x2, itp_mad_lin.(x1, x2')', xaxis=:log, xlabel="absolute elevation change over specified time period (m)", ylabel="surface elevation (m)", title="NMAD (-)")
        Plots.savefig(joinpath(fig_path,"nmad_interpolation.png"))
    end
    @printf("Kurtosis after standardization: %1.2f\n", kurtosis(df.dh_detrend))
    destandardize(dh, bin_field_1, bin_field_2) = dh .* std_y .* itp_mad_lin.(bin_field_1,bin_field_2) .+ itp_med_lin.(bin_field_1,bin_field_2) .+ mean_y
    return df, destandardize
end

function make_geotable(input::Vector, x::Vector, y::Vector)
    table    = (;Z = input)
    coords   = [(xi, yi) for (xi,yi) in zip(x,y)]
    geotable = georef(table, coords)
    return geotable
end

function fit_variogram(x::Vector{T}, y::Vector{T}, input::Vector{T}; nlags=90, maxlag=7e5, custom_var, param_cond, p0, fig_path="", sample_frac=1) where T <: Real
    println("Estimating variogram...")
    data = make_geotable(input, x, y)
    if sample_frac < 1
        nsamples = ceil(Int,sample_frac*length(x))
        smplr = UniformSampling(nsamples,replace=false) # replace=false ensures no point is sampled more than once
        data  = sample(data,smplr)
    end
    # compute empirical variogram
    U = data |> UniqueCoords()
    gamma = EmpiricalVariogram(U, :Z; estimator=:cressie, nlags,  maxlag) # for some reason the lsqfit has more problems fitting a good line for larger nlags
    # gi    = findall(gamma.ordinate .!= 0)   # sometimes there are γ=0 values, not sure why
    function get_γ(x, params)
        if !param_cond(params)  # some parameters are not accepted
            return -9999.0 .* ones(length(x))
        end
        f = custom_var(params)
        return f.(x)
    end
    # fit a covariance function
    ff = LsqFit.curve_fit(get_γ, gamma.abscissa, gamma.ordinate, p0);
    varg = custom_var(ff.param)

    # plot
    if !isempty(fig_path)
        Plots.scatter(gamma.abscissa .* 1e-3, gamma.ordinate, label="Empirical variogram", color=:black, markerstrokewidth=0, wsize=(1400,800), xlabel="Distance [km]", bottommargin=10Plots.mm, leftmargin=4Plots.mm)
        Plots.plot!([0;gamma.abscissa] .* 1e-3, varg.([0;gamma.abscissa]), label="Variogram fit", lw=2)
        Plots.savefig(joinpath(fig_path,"variogram.png"))
    end
    if p0 == ff.param @warn "Fitting of variogram failed, choose better initial parameters or reduce nlags." end
    return varg, ff
end

function generate_random_fields(output_dir; std_devs, corr_ls, x, y, destand, ir_random_field, rec, template_file, n_fields)
    k_m = 100.0
    nh  = 10000
    lx = x[end] - x[1]
    ly = y[end] - y[1]
    nx, ny = length(x), length(y)
    dest_files = Vector{String}(undef, n_fields)
    for i in 1:n_fields
        rftot = zeros(nx, ny)
        for (sf, rn) in zip(std_devs, corr_ls)
            cl = (rn, rn)
            rf = generate_grf2D(lx, ly, sf, cl, k_m, nh, nx, ny, cov_typ="expon", do_reset=false, do_viz=false);
            rftot .+= Array(rf)
        end
        # multiply with nmad and sigmas to get back variability w.r.t. binning variables
        rftot_destand = zeros(Float32, nx, ny)
        rftot_destand[ir_random_field] = destand(rftot[ir_random_field], ir_random_field)
        # smooth over a 5x5 pixel window
        rftot_smooth = mapwindow(median, rftot_destand, (5,5))
        # add the random field to the reconstruction
        rftot_smooth[ir_random_field]   .+= rec[ir_random_field]
        rftot_smooth[rftot_smooth .<= 0] .= no_data_value
        dest_files[i]  = joinpath(output_dir, "rec_rand_id_$(i).nc")
        save_netcdf(dest_files[i], template_file, [rftot_smooth], ["surface"], Dict("surface" => Dict{String,Any}()))
    end
    return dest_files
end


function prepare_random_sims(ref_file, bedm_file, obs_aero_file, obs_ATM_file, dhdt_file, mask_file;
                             atm_dh_dest_file, fig_path, custom_var, param_cond, p0, nbins1, nbins2, min_n_sample=100, blockspacing=5e3)
    # read in
    ref          = NCDataset(ref_file)["surface"][:]
    x            = NCDataset(ref_file)["x"][:]
    y            = NCDataset(ref_file)["y"][:]
    h_aero       = NCDataset(obs_aero_file)["Band1"][:]
    dhdt         = NCDataset(dhdt_file)["Band1"][:]
    glacier_mask = NCDataset(mask_file)["Band1"][:]
    bedm_mask    = NCDataset(bedm_file)["mask"][:]

    # indices
    I_no_ocean = findall(vec(.!ismissing.(glacier_mask) .&& (bedm_mask .!= 1)))
    idx_aero   = findall(vec(.!ismissing.(h_aero) .&& (h_aero  .> 0) .&& .!ismissing.(ref) .&& (bedm_mask .!= 1.0) .&& .!ismissing.(glacier_mask) .&& (abs.(ref .- h_aero ) .> 0.0)))

    # aerodem
    df_aero = get_aerodem_df(h_aero, ref, dhdt, x, y, idx_aero)

    # need a file with only one band for interp_points operation in xdem python;
    # plus need to reference to ellipsoid to be consistent with atm before differencing the two
    ref_surface_ellipsoid = get_surface_file(ref_file, bedm_file, remove_geoid=true)
    ref_surface_geoid     = get_surface_file(ref_file, bedm_file, remove_geoid=false)
    # interpolate GrIMP DEM on ATM points and calculate difference
    py_point_interp(ref_surface_ellipsoid, ref_surface_geoid, obs_ATM_file, atm_dh_dest_file)
    rm(ref_surface_ellipsoid)
    rm(ref_surface_geoid)     # delete them again as not needed anymore

    # block reduce with python package verde; average data that is heavily oversampled in direction of flight
    atm_dh_reduced = splitext(atm_dh_dest_file)[1]*"_reduced"*splitext(atm_dh_dest_file)[2]
    py_block_reduce(atm_dh_dest_file, atm_dh_reduced, blockspacing)

    # atm
    df_atm  = get_ATM_df(atm_dh_reduced, dhdt, x, y, df_aero; I_no_ocean)

    # merge aerodem and atm data
    df_all = vcat(df_aero, df_atm, cols=:intersect)

    # standardize, describing variance and bias as a function of dhdt and elevation
    df_all.dhdt = abs.(df_all.dhdt)
    df_all, destandardize = standardizing_2D(df_all, :dhdt, :h_ref; nbins1, nbins2, min_n_sample, fig_path);
    dhdt = replace_missing(dhdt, 0.0)   # the variance and bias linear interpolation function don't accept a type missing
    ref  = replace_missing(ref, 0.0)
    destand(dh, idx) = destandardize(dh, abs.(dhdt[idx]), ref[idx])

    # plot after standardizing
    Plots.scatter(df_all.x, df_all.y, marker_z=df_all.dh_detrend, label="", markersize=2.0, markerstrokewidth=0, cmap=:RdBu, clims=(-4,4), aspect_ratio=1, xlims=(-7e5,8e5), xlabel="Easting [m]", ylabel="Northing [m]", colorbar_title="[m]", title="Standardized elevation difference (GrIMP - historic)", grid=false, wsize=(1700,1800))
    Plots.savefig(joinpath(fig_path,"data_standardized.png"))
    Plots.scatter(df_all.x, df_all.y, marker_z=df_all.dh, label="", markersize=2.0, markerstrokewidth=0, cmap=:RdBu, clims=(-50,50), aspect_ratio=1, xlims=(-7e5,8e5), xlabel="Easting [m]", ylabel="Northing [m]", colorbar_title="[m]", title="dh non-standardized", grid=false, wsize=(1700,1800))
    Plots.savefig(joinpath(fig_path,"data_non-standardized.png"))

    # variogram
    varg, ff = fit_variogram(F.(df_all.x), F.(df_all.y), F.(df_all.dh_detrend); maxlag=1.5e6, nlags=200, custom_var, param_cond, sample_frac=0.05, p0, fig_path)
    return df_all, varg, ff, destand, I_no_ocean, idx_aero
end

function SVD_random_fields(rec_file::String, bedm_file::String, obs_aero_file::String, obs_ATM_file::String, dhdt_file::String, mask_file::String;
                           nbins1::Int=10, nbins2::Int=30,  # amount of bins for 2D standardization
                           n_fields::Int=10)                # number of simulations
    main_output_dir  = joinpath("output","SVD_RF")
    fig_path         = joinpath(main_output_dir, "figures")
    sims_path        = joinpath(main_output_dir, "simulations")
    atm_dh_dest_file = joinpath(dirname(obs_ATM_file), "SVD_rec_minus_atm.csv")
    mkpath(fig_path)
    mkpath(sims_path)

    # define variogram function for ParallelRandomFields
    custom_var(params) =   x ->
        params[1] .* (1 .-  exp.(-sqrt(2) * x./params[3])) .+
        params[2] .* (1 .-  exp.(-sqrt(2) * x./params[4]))
    param_cond(params) = all(params .> 0.0) # conditions on parameters
    # initial guess for parameters
    p0 = [0.5, 0.5, 1e4, 4e5]

    # standardize and get variogram
    _, varg, ff, destand, I_no_ocean, _ = prepare_random_sims(rec_file, bedm_file, obs_aero_file, obs_ATM_file, dhdt_file, mask_file;
                                                           atm_dh_dest_file, fig_path, custom_var, param_cond, p0, nbins1, nbins2)
    @printf("Sum of variances in variogram: %.2f \n", sum(ff.param[1:2]))
    @printf("Correlation length scales: %d and %d \n", ff.param[3], ff.param[4])

    # ParallelRandomFields
    std_devs   = sqrt.(ff.param[1:2] ./ sum(ff.param[1:2]))
    corr_ls    = ff.param[3:4]
    obs        = ncread(obs_aero_file, "Band1")
    rec        = ncread(rec_file,"surface")
    x          = ncread(rec_file,"x")
    y          = ncread(rec_file,"y")
    ir_random_field = findall(vec(obs .> 0 .|| rec .> 0))
    rf_files = generate_random_fields(sims_path; std_devs, corr_ls, x, y, destand, ir_random_field, rec, template_file=obs_aero_file, n_fields)
    return rf_files
end

# for validation
function step_through_folds(flds, evaluate_fun, geotable; save_distances=false, save_coords=false)
    dif_blocks     = [Float64[] for i in flds]
    if save_distances
        NNdist_blocks  = [Float64[] for i in flds]
    end
    if save_coords
        xcoord_blocks  = [Float64[] for i in flds]
        ycoord_blocks  = [Float64[] for i in flds]
    end
    for (j,fs) in enumerate(flds)
        # find the neighbors that the folds routine (https://github.com/JuliaEarth/GeoStatsBase.jl/blob/master/src/folding/block.jl) leaves out
        # there might be a mistake in the partitioning routine in Meshes.jl, the neighbors don't make sense (also not tested well)
        neighbors = Vector(1:length(geotable.geometry))
        deleteat!(neighbors, unique(sort([fs[1];fs[2]])))
        append!(fs[1],neighbors)

        sdat  = view(geotable, fs[1])
        stest = view(domain(geotable), fs[2])
        @assert length(sdat.Z) > length(stest)

        y_pred = evaluate_fun(fs[1],fs[2])
        dif_blocks[j] = y_pred .- geotable.Z[fs[2]]
        # Plots.scatter!(x_[fs[1]], y_[fs[1]], markersize=2, color="grey",markerstrokewidth=0)
        # Plots.scatter!(x_[fs[2]], y_[fs[2]], markersize=2, markerstrokewidth=0)
        # if j == 1
        #     break
        # end
        if save_distances
            # save distance of each test point to closest training/data point
            ids_nn  = zeros(Int,length(stest))
            dist_nn = zeros(length(stest))
            for (i,st) in enumerate(stest)
                id_st, dist = searchdists(st, KNearestSearch(domain(sdat),1))
                ids_nn[i]  = id_st[1]
                dist_nn[i] = dist[1]
            end
            NNdist_blocks[j] = dist_nn
        end
        if save_coords
            # save coordinates of the test points
            crds = coordinates.(stest)
            xcoord_blocks[j] = first.(crds)
            ycoord_blocks[j] = last.(crds)
        end
    end
    rt = (vcat(dif_blocks...))
    if save_distances
        rt = (rt, vcat(NNdist_blocks...))
    end
    if save_coords
        rt = (rt, vcat(xcoord_blocks...), vcat(ycoord_blocks...))
    end
    return rt
end

function generate_sequential_gaussian_sim(output_geometry, geotable_input, varg; n_fields, maxn)
    method  = SEQMethod(maxneighbors=maxn)
    process = GaussianProcess(varg)
    tic   = Base.time()
    sims    = rand(process, output_geometry, geotable_input, n_fields, method)
    toc   = Base.time() - tic
    @printf("SGS took %d minutes. \n", toc / 60)
    return sims
end

function do_kriging(output_geometry::Domain, geotable_input::AbstractGeoTable, varg::Variogram; maxn::Int)
    model  = Kriging(varg)
    tic    = Base.time()
    interp = geotable_input |> InterpolateNeighbors(output_geometry, model, maxneighbors=maxn)
    toc    = Base.time() - tic
    @printf("Kriging took %d minutes. \n", toc / 60)
    return interp, toc
end

function do_destandardization(grimp_file::String, bedm_file::String, obs_aero_file::String, obs_ATM_file::String, dhdt_file::String, mask_file::String,
                              fig_path::String, atm_dh_dest_file::String;
                              nbins1::Int, nbins2::Int, blockspacing)  # amount of bins for 2D standardization

    # define variogram function to fit
    custom_var(params) = SphericalVariogram(range=params[1], sill=params[4]) + # ./ sum(params[4:6])) +
                         SphericalVariogram(range=params[2], sill=params[5]) + # ./ sum(params[4:6])) +
                         SphericalVariogram(range=params[3], sill=params[6]) # ./ sum(params[4:6]))
    param_cond(params) = all(params .> 0) # && all(0.0 .< params[1:3] .< 1.0)
    # initial guess for parameters
    p0 = [1e4, 5e5, 1.4e6, 0.3, 0.3, 0.3]

    # standardize and get variogram
    df_all, varg, ff, destand, I_no_ocean, idx_aero = prepare_random_sims(grimp_file, bedm_file, obs_aero_file, obs_ATM_file, dhdt_file, mask_file;
                                                                         atm_dh_dest_file, fig_path, custom_var, param_cond, p0, nbins1, nbins2, blockspacing)
    # force overall variance of variogram to be one
    @printf("Sum of variances in variogram: %.2f \n", sum(ff.param[4:6]))
    ff.param[4:6] .= ff.param[4:6] ./ sum(ff.param[4:6])
    varg           = custom_var(ff.param)
    display(varg)

    ir_sim      = setdiff(I_no_ocean, idx_aero)  # indices that are in I_no_ocean but not in idx_aero
    x           = NCDataset(obs_aero_file)["x"][:]
    y           = NCDataset(obs_aero_file)["y"][:]
    grid_output = PointSet([Point(xi,yi) for (xi,yi) in zip(x[get_ix.(ir_sim, length(x))], y[get_iy.(ir_sim, length(x))])])
    geotable    = make_geotable(df_all.dh_detrend, df_all.x, df_all.y)

    # prepare predicted field, fill with aerodem observations where available
    h_aero               = NCDataset(obs_aero_file)["Band1"][:]
    h_grimp              = NCDataset(grimp_file)["surface"][:]
    h_grimp              = replace_missing(h_grimp, 0.0)
    h_predict            = zeros(size(h_aero))
    h_predict[idx_aero] .= h_aero[idx_aero]

    function get_predicted_field(dh)
        @assert length(dh) == length(ir_sim)
        h_predict[ir_sim]           .= h_grimp[ir_sim] .- destand(dh, ir_sim)
        h_predict[h_predict .<= 0.] .= no_data_value
        return h_predict
    end
    return grid_output, geotable, varg, get_predicted_field, ir_sim
end

function geostats_interpolation(grimp_file::String, bedm_file::String, obs_aero_file::String, obs_ATM_file::String, dhdt_file::String, mask_file::String;
                                nbins1::Int=6, nbins2::Int=10,  # amount of bins for 2D standardization
                                maxn::Int,                      # maximum neighbors for interpolation method
                                method::Symbol=:kriging,        # either :kriging or :sgs
                                n_fields::Int=10)               # number of simulations in case of method=:sgs
    main_output_dir  = joinpath("output","geostats_interpolation")
    fig_path         = joinpath(main_output_dir, "figures/")
    atm_dh_dest_file = joinpath(dirname(obs_ATM_file), "grimp_minus_atm.csv")
    mkpath(fig_path)

    grid_output, geotable, varg, get_predicted_field, ir_sim = do_destandardization(grimp_file, bedm_file, obs_aero_file, obs_ATM_file, dhdt_file, mask_file, fig_path, atm_dh_dest_file; nbins1, nbins2)
    ## !! ToDo !! use get_predicted_field function below

    if method == :sgs  # sequential gaussian simulations
        output_path = joinpath(main_output_dir, "SEQ_simulations/")
        mkpath(output_path)
        println("Generating sequential gaussian simulations...")
        sims = generate_sequential_gaussian_sim(grid_output, geotable, varg; n_fields, maxn)
        dest_files = Vector{String}(undef, n_fields)
        for (i,s) in enumerate(sims)
            h_predict[ir_sim]           .= h_grimp[ir_sim] .- destand(s.Z, ir_sim)
            h_predict[h_predict .<= 0.] .= no_data_value
            dest_files[i]  = joinpath(output_path, "rec_sgs_id_$(i).nc")
            save_netcdf(dest_files[i], obs_aero_file, [h_predict], ["surface"], Dict("surface" => Dict{String,Any}()))
        end
        return dest_files
    elseif method == :kriging
        output_path = joinpath(main_output_dir, "kriging/")
        mkpath(output_path)
        println("Kriging...")
        interp = do_kriging(grid_output, geotable, varg; maxn)
        h_predict[ir_sim]           .= h_grimp[ir_sim] .- destand(interp.Z, ir_sim)
        h_predict[h_predict .<= 0.] .= no_data_value
        dest_file                    = joinpath(output_path, "rec_kriging.nc")
        save_netcdf(dest_file, obs_aero_file, [h_predict], ["surface"], Dict("surface" => Dict{String,Any}()))
        return dest_file
    end
end
