# tasks from the chou
# subsample to 5'000 points on entire ice sheet
# use erode, get variogram of upper band of aerodem, is there a 500k correlation length?

using svd_IceSheetDEM
using NCDatasets, Interpolations, DataFrames, CSV, ProgressMeter, GeoStats, LsqFit, ParallelStencil, ImageMorphology, ImageSegmentation, JLD2
using StatsBase
import Plots

@init_parallel_stencil(Threads, Float64, 1)

gr = 1200

get_ix(i,nx) = i % nx == 0 ? nx : i % nx
get_iy(i,nx) = cld(i,nx)

# dhdt
# dhdt = NCDataset("data/dhdt/CCI_GrIS_RA_SEC_5km_Vers3.0_2021-08-09_g600_1994-2010.nc")["Band1"][:,:]
# dhdt[ismissing.(dhdt)] .= NaN

# masks
ds_mask = NCDataset("data/gris-imbie-1980/imbie_mask_g$(gr).nc")["Band1"][:]
bedm_mask = NCDataset("data/bedmachine/bedmachine_g$(gr).nc")["mask"][:]

# GrIMP
# grimp = NCDataset("data/bedmachine/bedmachine_g$(gr).nc")["surface"][:,:]
grimp = NCDataset("data/grimp/surface/grimp_geoid_corrected_g$(gr).nc")["surface"][:,:]

# aero
# ds_aero      = NCDataset("data/aerodem/aerodem_rm-filtered_geoid-corr_g$(gr).nc")
# h_aero_all      = ds_aero["Band1"][:,:]
# dh_aero_all = zeros(size(h_aero))
# idx_aero = findall(.!ismissing.(vec(h_aero_all)) .&& .!ismissing.(vec(ds_mask)) .&& vec((bedm_mask .!= 1)) .&& .!ismissing.(vec(grimp)))
# dh_aero_all[idx_aero] = grimp[idx_aero] .- h_aero_all[idx_aero]


ds_aero = NCDataset("data/newgrimp_minus_aero_g$(gr).nc")
dh_aero_all = ds_aero["dh"][:,:,1]
dh_aero_all = dh_aero_all[:,end:-1:1]
x  = sort(ds_aero["x"])
y  = sort(ds_aero["y"])
idx_aero = findall(.!ismissing.(vec(dh_aero_all)) .&& .!ismissing.(vec(ds_mask)) .&& vec((bedm_mask .!= 1)))
df_aero = DataFrame(:x       => x[get_ix.(idx_aero, length(x))],
                        :y       => y[get_iy.(idx_aero, length(x))],
                        :h_grimp => grimp[idx_aero],
                        :dh      => dh_aero_all[idx_aero],
                        :idx     => idx_aero,
                        :source .=> :aerodem )

# belt
# dh_aero_mat = falses(length(x),length(y))
# dh_aero_mat[df_aero.idx] .= true
# dh_eroded = erode(dh_aero_mat)

# grimp_mat = falses(length(x),length(y))
# idx_gg = findall(.!ismissing.(vec(ds_mask)) .&& vec((bedm_mask .!= 1)))
# grimp_mat[idx_gg] .= true
# erode!(grimp_mat, r=4)

# i_belt = findall(vec(grimp_mat) .& .!vec(dh_eroded) .& vec(dh_aero_mat))

# dh_belt = falses(length(x),length(y))
# dh_belt[i_belt] .= true
# Plots.heatmap(dh_belt')
# df_aero_belt = DataFrame(:x       => x[get_ix.(i_belt, length(x))],
#                         :y       => y[get_iy.(i_belt, length(x))],
#                         :h_grimp => grimp[i_belt],
#                         :dh      => dh_aero_all[i_belt],
#                         :idx     => i_belt)
# df_aero_belt.dh_detrend = (df_aero_belt.dh .- median(df_aero_belt.dh)) ./ mad(df_aero_belt.dh)

# atm
df_atm = CSV.read("data/grimp/surface/atm_new_grimp_interp_pts_no_registr.csv", DataFrame)
df_atm[!,:source] .= :atm
mindist = 5e4 # minimum distance between aerodem and atm points
function choose_atm(x, y)::Bool
    dist_to_aero = minimum(pairwise(Distances.euclidean, [x y], [df_aero.x df_aero.y], dims=1)[:])
    is_above_mindist = dist_to_aero > mindist
    return is_above_mindist
end
println("selecting flightline values...")
id = sort(StatsBase.sample(1:size(df_atm,1), 100000, replace=false))  # without pre-selecting a subsample of points the filtering takes a long time
keepat!(df_atm, id)
filter!([:x,:y] => choose_atm, df_atm)
# remove outliers --> do it after standardization
# atm_to_delete = findall(abs.(df_atm.dh) .> 5 .* mad(df_atm.dh))
# deleteat!(df_atm, atm_to_delete)
# id2 = sort(StatsBase.sample(1:size(df_atm,1), cld(length(df_aero.dh),2), replace=false))
# keepat!(df_atm, id2)

# merge aerodem and atm data
df_all = vcat(df_aero, df_atm, cols=:intersect)

# plot
dh_plot = Array{Union{Float32,Missing}}(missing, (length(x),length(y)))
dh_plot[df_aero.idx] = df_aero.dh
Plots.heatmap(x,y, dh_plot', cmap=:bwr, clims=(-20,20))
Plots.scatter!(df_atm.x, df_atm.y, marker_z=df_atm.dh, color=:bwr, markersize=0.5, markerstrokewidth=0, legend=false)
Plots.savefig("dplot.png")

# mean trend as a fct of elevation
dh_binned, ELV_bin_centers = svd_IceSheetDEM.bin_equal_sample_size(df_all.h_grimp, df_all.dh, 10000)  # 16000
itp_bias = linear_interpolation(ELV_bin_centers, median.(dh_binned),  extrapolation_bc = Interpolations.Flat())
Plots.scatter(ELV_bin_centers, median.(dh_binned))
ELV_plot = minimum(df_all.h_grimp):10:maximum(df_all.h_grimp)
Plots.plot!(ELV_plot, itp_bias(ELV_plot), label="linear interpolation", legend=:bottomright)
Plots.savefig("c1plot.png")

# standard deviation as a fct of elevation
itp_std = linear_interpolation(ELV_bin_centers, mad.(dh_binned),  extrapolation_bc = Interpolations.Flat())
Plots.scatter(ELV_bin_centers, mad.(dh_binned))
ELV_plot = minimum(df_all.h_grimp):10:maximum(df_all.h_grimp)
Plots.plot!(ELV_plot, itp_std(ELV_plot), label="linear interpolation", legend=:topright)
Plots.savefig("c2plot.png")

# standardize
df_all.dh_detrend = (df_all.dh .- itp_bias(df_all.h_grimp)) ./ itp_std(df_all.h_grimp)

# remove outliers after standardizing
all_to_delete = findall(abs.(df_all.dh_detrend) .> 3 .* mad(df_all.dh_detrend))
deleteat!(df_all, all_to_delete)
std_dh_detrend        = std(df_all.dh_detrend)
df_all.dh_detrend ./= std_dh_detrend

Plots.density(df_all.dh_detrend, label="Standardized observations", xlims=(-10,10))
# Plots.density!((df_all.dh .- mean(df_all.dh)) ./ std(df_all.dh), label="Standardized without binning")
Plots.plot!(Normal(), label="Normal distribution")
Plots.savefig("eplot.png")


# plot again after standardizing
Plots.scatter(df_all.x, df_all.y, marker_z=df_all.dh_detrend, color=:bwr, markersize=0.5, markerstrokewidth=0, cmap=:bwr, clims=(-3,3))
Plots.savefig("d2plot.png")

# variogram
println("Variogram...")
df_varg = df_all
table_all  = (; Z=df_varg.dh_detrend)
coords_all = [(df_varg.x[i], df_varg.y[i]) for i in 1:length(df_varg.x)]
data       = georef(table_all,coords_all)
gamma      = EmpiricalVariogram(data, :Z; nlags=400,  maxlag=7e5, estimator=:cressie)
# fit a covariance function
function custom_var(x, params)
    if any(params[[1,3,5]] .>= 8e5)
        return -9999.0 .* ones(length(x))
    end
    γ1 = SphericalVariogram(range=params[1], sill=params[2])      # forcing nugget to be zero, not the case with the built-in fit function of GeoStats
    γ2 = SphericalVariogram(range=params[3], sill=params[4])
    γ3 = SphericalVariogram(range=params[5], sill=params[6]) #, nugget=params[9])
    f = γ1.(x) .+ γ2.(x) .+ γ3.(x)
    return f
end
ff = LsqFit.curve_fit(custom_var, gamma.abscissa, gamma.ordinate, [1e5, 0.2, 1e4, 0.2, 5e5, 0.5]);
# julia> ff.param
# 2-element Vector{Float64}:
#  50280.021902654196
#      0.94295155705642
# varg = ExponentialVariogram(range=ff.param[1], sill=ff.param[2], nugget=ff.param[3])
varg = SphericalVariogram(range=ff.param[1], sill=ff.param[2]) +
       SphericalVariogram(range=ff.param[3], sill=ff.param[4]) +
       SphericalVariogram(range=ff.param[5], sill=ff.param[6])
Plots.scatter(gamma.abscissa, gamma.ordinate, label="all observations")
Plots.plot!(gamma.abscissa,custom_var(gamma.abscissa, ff.param), label="LsqFit fit", lw=2)
Plots.savefig("vplot.png")

# varg = ExponentialVariogram(range=50280.021902654196, sill=0.94295155705642, nugget=0.05)   # nugget is uncertainty of observations

# only take a random selection of point to speed up Kriging
i_krig = sort(StatsBase.sample(1:length(df_all.dh_detrend), cld(length(df_all.x),3), replace=false))
df_krig = df_all[i_krig,:]
Plots.scatter(df_krig.x, df_krig.y, marker_z=df_krig.dh_detrend, color=:bwr, markersize=0.5, markerstrokewidth=0, cmap=:bwr, clims=(-3,3))


# table_box  = (; Z=Float32.(df_krig.dh_detrend))
# coords_box = [(xi,yi) for (xi,yi) in zip(df_krig.x,  df_krig.y)]
# geotable   = georef(table_box,coords_box)
# model      = Kriging(varg, 0.0)
# sk         = GeoStatsModels.fit(model, geotable)

@parallel_indices (ibox) function interpolate_subsets!(mb, input_dh, input_x, input_y, output_x, output_y, input_gr)
    if 1 <= ibox <= length(output_x)
        println(ibox)
        # extract data points in sub-area
        x_min, x_max = extrema(output_x[ibox])
        y_min, y_max = extrema(output_y[ibox])

        # prepare data and grid for interpolation
        table_input  = (; Z=Float32.(input_dh[ibox]))
        coords_input = [(xi,yi) for (xi,yi) in zip(input_x[ibox],  input_y[ibox])]
        geotable_input   = georef(table_input,coords_input)
        model      = Kriging(varg, 0.0)

        grid_output       = CartesianGrid((x_min-input_gr/2, y_min-input_gr/2), (x_max+input_gr/2, y_max+input_gr/2), (input_gr, input_gr))

        # do interpolation
        # sk = GeoStatsModels.fit(model, geotable)
        # skdist =[ GeoStatsModels.predictprob(sk, :Z, Point(xi, yi)) for xi in x_min:gr:x_max, yi in y_min:gr:y_max]
        # mbi = mean.(skdist)
        # mb[ibox,:,:] = mbi
        interp = geotable_input |> Interpolate(grid_output, model) #, minneighbors=300, maxneighbors=500)
        mb[ibox,:,:] = reshape(interp.Z, size(grid_output))
    end
    return
end

# x_i1 = 110:450
# y_i1 = 1550:1800
# # 3.6 hours

# mb_1 = interpolate_subset(df_all, x_i1, y_i1)

# x_i2 = 170:510
# y_i2 = 1550:1800

# mb_2 = interpolate_subset(df_all, x_i2, y_i2)

# outputs_ix = [101:150, 151:200, 201:250, 251:300, 301:350, 351:400, 401:450] #, 170:340, 230:400, 290:460, 350:520, 410:580, 470:640, 530:700]
# outputs_iy = [1600:1650] #, 1550:1670, 1550:1670, 1550:1670, 1550:1670, 1550:1670, 1550:1670, 1550:1670]

x00 = 1
step = 120
x0s   = range(x00; step, stop=length(x))
xends = range(x00+step-1; step, stop=length(x)+step)
outputs_ix = [range(x0s[i], xends[i]) for i in eachindex(x0s)]

y0s   = range(x00; step, stop=length(y))
yends = range(x00+step-1; step, stop=length(y)+step)
outputs_iy = [range(y0s[i], yends[i]) for i in eachindex(y0s)]


# outputs_iy  = repeat([1600:1650], length(outputs_ix))
# x_is = repeat([100:250], 100)
# y_is = repeat([1500:1800], 100)


input_x = []
input_y = []
output_x = []
output_y = []
input_dh = []
input_h  = []
for iy in outputs_iy
    for ix in outputs_ix
        dmargin = Int(div(maximum(range.(varg.γs)), gr, RoundUp))
        x_min = x[max(1, ix[1]-dmargin)]
        y_min = y[max(1, iy[1]-dmargin)]
        x_max = x[min(length(x), ix[end]+dmargin)]
        y_max = y[min(length(y), iy[end]+dmargin)]
        # x_min, x_max = extrema(x[ix])
        # y_min, y_max = extrema(y[iy])
        function is_in_box(x, y)::Bool
            is_in_x = x_min <= x <= x_max
            is_in_y = y_min <= y <= y_max
            return is_in_x && is_in_y
        end
        # i_krig = sort(unique(rand(1:length(df_all.dh_detrend), 20000)))
        # df_krig = df_all[i_krig,:]
        df_box = filter([:x,:y] => is_in_box, df_krig)
        # df_aero_box = filter([:x,:y] => is_in_box, df_aero)
        # df_aero_box.dh_detrend = (df_aero_box.dh .- itp_bias(df_aero_box.h_grimp)) ./ itp_std(df_aero_box.h_grimp)
        # iix = [findfirst(i .== x[x_i]) for i in df_aero_box.x]
        # iiy = [findfirst(i .== y[y_i]) for i in df_aero_box.y]
        push!(input_x,  df_box.x)
        push!(input_y,  df_box.y)
        push!(input_h,  df_box.h_grimp)
        push!(input_dh, df_box.dh_detrend)
        push!(output_x, x[ix])
        push!(output_y, y[iy])
    end
end

#
# grimp_target_file = "data/grimp/surface/grimp_geoid_corrected_g5000.nc"
# grimp_target = NCDataset(grimp_target_file)
# output_gr = diff(grimp_target["x"])[1]
# x_min, x_max = extrema(grimp_target["x"][:])
# y_min, y_max = extrema(grimp_target["y"][:])
# cg = CartesianGrid((x_min-gr/2, y_min-gr/2), (x_max+gr/2, y_max+gr/2), (output_gr, output_gr))
#

println("Starting kriging...")
mb   = @zeros(length(outputs_ix)*length(outputs_iy), length(outputs_ix[1]), length(outputs_iy[1]))
tic = Base.time()
@parallel interpolate_subsets!(mb, input_dh, input_x, input_y, output_x, output_y, Float64.(gr))
toc = Base.time() - tic
tt = toc / 60
println("Interpolation took $tt minutes.")

# stitch back together
mb_full = zeros(Float32, length(outputs_ix)*step, length(outputs_iy)*step)
ib = 1
for iy in outputs_iy
    for ix in outputs_ix
        iix = ix .- outputs_ix[1][1] .+ 1
        iiy = iy .- outputs_iy[1][1] .+ 1
        mb_full[iix,iiy] = mb[ib,:,:]
        ib += 1
    end
end

# save
save("krigoutput.jld2", Dict("mb_full" => mb_full, "mb" => mb))



# mb_all = vcat([mb[i,:,:] for i in 1:12]...)
# Plots.heatmap(mb_all', cmap=:bwr, clims=(-3,3))
# Plots.savefig("atry.png")

# dsts = [evaluate(Euclidean(), (x_boxes[1][j],y_boxes[1][j]), (x_boxes[1][i],y_boxes[1][i])) for i in 1:length(x_boxes[1]), j in 1:length(x_boxes[1])]  # calculate the distances between points
# vdst = vec(UpperTriangular(dsts))
# idst = findall(vdst .!= 0)  # diagonal entries are zero
# Plots.histogram(vdst[idst], nbins=50)




# ttocs_all
# 5-element Vector{Float64}:
#   0.09029412269592285
#   0.9920129776000977
#   4.69612193107605
#  16.105940103530884
#  40.58135485649109
# Plots.plot(length.(x_is).*length.(y_is), ttocs_all, xscale=:log10)
# Plots.plot(length.(x_is).*length.(y_is), ttocs_all)


# for i_krig ... 3500
# output_gr = 5000 -> 14 minutes
# output_gr = 4000 -> 21 minutes
# output_gr = 3000 -> 40 minutes (I think)

# for i_krig ... 5000
# output_gr = 5000 -> 29 minutes

# for i_krig ... 9000
# output_gr = 5000 -> 101 minutes


## destandardize
# grimp_tg = copy(grimp)
# grimp_tg[ismissing.(grimp_tg)] .= NaN
# h_predict_all = zeros(length(x),length(y))
# h_predict_all[outputs_ix[1],outputs_iy[1]] = grimp_tg[outputs_ix[1],outputs_iy[1]] .- (mb[1,:,:] .*std_dh_detrend.*itp_std.(grimp_tg[outputs_ix[1],outputs_iy[1]]) .+ itp_bias.(grimp_tg[outputs_ix[1],outputs_iy[1]]))
# h_predict_all[h_predict_all .<= 0 .|| isnan.(h_predict_all)] .= no_data_value
# svd_IceSheetDEM.save_netcdf("output/kriging_g1200.nc", "data/grimp/surface/grimp_geoid_corrected_g$(gr).nc", [h_predict_all], ["surface"], Dict("surface" => Dict{String, Any}()))
# gdalwarp("output/kriging_g5000.nc"; gr=600, srcnodata="-9999", dstnodata="-9999", dest="output/kriging_g600_warped_from_5000.nc")



## Plotting stuff


# Plots.heatmap(x[outputs_ix[1]], y[outputs_iy[1]], mb[1,:,:]', cmap=:bwr, clims=(-3,3), aspect_ratio=1)
# p1 = Plots.heatmap(mb[1,:,:]', cmap=:bwr, clims=(-4,4), aspect_ratio=1)
# p2 = Plots.heatmap(mb[2,:,:]', cmap=:bwr, clims=(-1,1), aspect_ratio=1)
# Plots.plot(p1,p2)

# overlap_1 = mb[1,61:end,:]
# overlap_2 = mb[2,1:end-60,:]
# p1 = Plots.heatmap(x_is[1][61:end],y_is[1],overlap_1', cmap=:bwr, clims=(-1,1), aspect_ratio=1)
# p2 = Plots.heatmap(x_is[2][1:end-60],y_is[2],overlap_2', cmap=:bwr, clims=(-1,1), aspect_ratio=1)
# Plots.plot(p1,p2)
# Plots.heatmap(x[x_is[1][61:end]], y[y_is[1]],overlap_1' .- overlap_2', cmap=:bwr, clims=(-1.0,1.0))
# Plots.heatmap(overlap_1' .- overlap_2', cmap=:bwr, clims=(-1e-1,1e-1))
# Plots.savefig("diff_overlap.png")

# # geotable, grid, model, xis, yis, grimp, itp_bias, itp_std, idx_aero = prepare(gr)
# # nx, ny = length(xis), length(yis)
# # mb = do_kriging(geotable, grid, model, nx,ny)

# h_predict = grimp[x_i,y_i] .- (mb.*itp_std.(grimp[x_i,y_i]) .+ itp_bias.(grimp[x_i,y_i]))
# Plots.heatmap(x[x_i], y[y_i], h_predict', aspect_ratio=1, clims=(0,1600))
# # Plots.savefig("h_predict.png")

# h_aero = grimp .- dh_aero_all
# idx_box = findall(.!ismissing.(h_aero[x_i,y_i]))


# dd = Array{Union{Float32,Missing}}(missing, (length(x_i),length(y_i)))
# dd[idx_box_dhdata] = h_predict[idx_box_dhdata] .- h_aero[x_i,y_i][idx_box_dhdata]
# Plots.heatmap(x[x_i], y[y_i], dd', aspect_ratio=1, cmap=:bwr, clims=(-60,60))
# Plots.savefig("diff_to_aero.png")

# Plots.heatmap(x[x_i],y[y_i],h_aero[x_i,y_i]', aspect_ratio=1, clims=(0,1500))
# hh = Array{Union{Float32,Missing}}(missing, (length(x_i),length(y_i)))
# hh[idx_box] = h_predict[idx_box]
# Plots.heatmap(x[x_i],y[y_i],hh', aspect_ratio=1, clims=(0,1500))


# # x_min, x_max = extrema(x[x_i])
# # y_min, y_max = extrema(y[y_i])
# # df_aero_box = filter([:x,:y] => is_in_box, df_aero)
# db = Array{Union{Float32,Missing}}(missing, (length(x),length(y)))
# db[df_aero_box.idx] .= df_aero_box.dh
# Plots.heatmap(x[x_i], y[y_i],reshape(db,length(x),length(y))[x_i,y_i]', cmap=:bwr, clims=(-5,5), aspect_ratio=1)
# idx_box_dhdata = findall(.!ismissing.(reshape(db,length(x),length(y))[x_i,y_i]))

# mbpart = zeros(length(x_i),length(y_i))
# mbpart[idx_box_dhdata] .= mb[idx_box_dhdata]
# Plots.heatmap(x[x_i], y[y_i],mbpart', cmap=:bwr, clims=(-5,5), aspect_ratio=1)
